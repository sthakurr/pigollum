"""
PiGollumOptimizer: Principle-guided Bayesian Optimization.

Architecture
────────────
    Warm-start (once)
    ─────────────────
    1. LLM generates broad domain-knowledge principles (no data)
    2. LLM refines each principle with supporting/contradicting evidence from training set
    → Result: grounded, task-level seed principles

    Per BO iteration
    ────────────────
    1. Fine-tune LLM-GP on new data point
    2. Compute GP predictions (means + stds) for all candidates
    3. Compute acquisition scores (GP mean / UCB / std based on action type)
    4. 3-agent pipeline (if enabled):
       a. Planner — takes (pk, yk): principle closest to oracle candidate + oracle output;
          uses per-principle exploration/exploitation scores to re-rank all principles
          and assign EXPLORE / VALIDATE / REFINE actions (PiFlow Appendix Q format)
       b. Hypothesis — generates directional hypothesis from action-annotated principles
          (PiFlow Appendix Q format)
       c. Scorer — scores top-k candidates (by acquisition score) against hypothesis;
          hybrid score = α * scorer_score + (1-α) * acq_score
    5. Greedy selection on hybrid scores → next candidate for oracle evaluation
    6. Oracle evaluation → extract principle from (sequence, outcome) → add to buffer

The ``principle_weight`` (α) can be scheduled to increase as more principles
accumulate, gradually shifting trust from pure BO toward principle guidance.
"""
import logging
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from torch import Tensor

# gollum package lives alongside pigollum under src/; no extra path needed

from gollum.bo.optimizer import BotorchOptimizer

from pigollum.principle.buffer import Principle, PrincipleBuffer
from pigollum.principle.extractor import PrincipleExtractor
from pigollum.principle.scorer import PrincipleScorer
from pigollum.principle.journal import PrincipleJournal
from pigollum.utils.sequence_utils import describe_sequence

logger = logging.getLogger(__name__)


class PiGollumOptimizer(BotorchOptimizer):
    """
    Principle-guided BO optimizer.

    Parameters
    ----------
    principle_weight : float
        α ∈ [0, 1].  Weight of principle scores relative to BO acquisition scores.
    min_principles_for_guidance : int
        Minimum number of principles before principle guidance activates.
    principle_weight_schedule : str | None
        Schedule for α: None/"constant", "linear", or "step".
    extractor : PrincipleExtractor | None
        LLM-based hypothesis and principle extraction.
    scorer : PrincipleScorer | None
        Exploration-exploitation scorer.
    buffer : PrincipleBuffer | None
        Pre-populated buffer.
    enable_post_acq_agents : bool
        Whether to run the 3-agent pipeline (Planner → Hypothesis → Scorer)
        after computing acquisition scores and before greedy selection.
    top_k_for_rescoring : int
        Number of top candidates (by acquisition score) passed to the Scorer agent.
    include_experimental_data : bool
        Whether to pass oracle history to the Hypothesis agent.
    All other kwargs are forwarded to BotorchOptimizer.
    """

    def __init__(
        self,
        *args,
        principle_weight: float = 0.3,
        min_principles_for_guidance: int = 3,
        principle_weight_schedule: Optional[str] = None,
        extractor: Optional[PrincipleExtractor] = None,
        scorer: Optional[PrincipleScorer] = None,
        buffer: Optional[PrincipleBuffer] = None,
        enable_post_acq_agents: bool = False,
        top_k_for_rescoring: int = 20,
        include_experimental_data: bool = True,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.principle_weight = principle_weight
        self.min_principles_for_guidance = min_principles_for_guidance
        self.principle_weight_schedule = principle_weight_schedule
        self.enable_post_acq_agents = enable_post_acq_agents
        self.top_k_for_rescoring = top_k_for_rescoring
        self.include_experimental_data = include_experimental_data

        self.principle_buffer = buffer or PrincipleBuffer()
        self.principle_extractor = extractor  # may be None
        self.principle_scorer = scorer or PrincipleScorer(device=str(self.tkwargs["device"]))
        self.journal: Optional[PrincipleJournal] = None

        self._iteration = 0
        self._last_action_info: dict = {}
        self._last_gp_means: Optional[np.ndarray] = None
        self._last_gp_stds: Optional[np.ndarray] = None
        self._oracle_history: List[Dict] = []       # accumulates across all iterations
        # pk/yk: principle closest to last oracle candidate + its measured outcome
        self._last_oracle_principle: Optional[Principle] = None
        self._last_oracle_y: Optional[Dict] = None
        # Last post-acquisition agent outputs — exposed for wandb / external loggers
        self._last_planner_response: Optional[str] = None
        self._last_direction_hyp: Optional[str] = None
        self._last_scorer_response: Optional[str] = None

    # ------------------------------------------------------------------
    # GP prediction helper (experiment agent)
    # ------------------------------------------------------------------

    def predict_with_gp(
        self,
        design_space: Tensor,
        train_x: Tensor,
        train_y: Tensor,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Get GP mean and std predictions for all candidates.

        This is the core "experiment agent" functionality: the GP provides
        predictions that validate/invalidate LLM hypotheses.

        Returns
        -------
        means : np.ndarray (M, n_objectives)
        stds : np.ndarray (M, n_objectives)
        """
        # DeepGP stores raw tokens as train_inputs and runs the featurizer
        # internally in forward(). We must pass raw tokens here — not
        # pre-projected embeddings — otherwise GPyTorch's cat(train, test)
        # will fail with a dimension mismatch.
        with torch.no_grad():
            means, variances = self.surrogate_model.predict(
                design_space.to(**self.tkwargs)
            )
            means = means.cpu().numpy()
            stds = variances.sqrt().cpu().numpy()

        return means, stds

    # ------------------------------------------------------------------
    # Core BO iteration: GP → 3-agent pipeline → greedy select
    # ------------------------------------------------------------------

    def suggest_next_experiments(
        self,
        train_x: Tensor,
        train_y: Tensor,
        design_space: Tensor,
        test: bool = False,
        candidate_sequences: Optional[List[str]] = None,
        objective_names: Optional[List[str]] = None,
    ):
        """
        Per-iteration pipeline:
        1. Fine-tune LLM-GP on (train_x, train_y)
        2. GP predicts means + stds for all candidates
        3. Compute acquisition scores (GP mean / UCB / std based on action type)
        4. 3-agent pipeline (if enable_post_acq_agents):
           a. Planner(pk, yk) — re-ranks principles with per-principle actions
           b. Hypothesis — directional hypothesis from action-annotated principles
           c. Scorer — re-ranks top-k candidates; hybrid score combines acq + LLM
        5. Greedy selection on hybrid scores

        Falls back to standard BO if no sequences or extractor are provided.
        """
        has_sequences = (
            candidate_sequences is not None
            and len(candidate_sequences) == design_space.size(0)
        )

        train_x = train_x.to(**self.tkwargs)
        train_y = train_y.to(**self.tkwargs)
        design_space = design_space.to(**self.tkwargs)

        if not test:
            self.train_surrogate_model(train_x, train_y)

        if not has_sequences or self.principle_extractor is None:
            logger.info("PiGollumOptimizer: using pure BO (no sequences or extractor).")
            return super().suggest_next_experiments(train_x, train_y, design_space, test=True)

        # ----------------------------------------------------------------
        # Step 1: GP predictions for all candidates
        # ----------------------------------------------------------------
        logger.info("PiGollumOptimizer: GP predicting all candidates…")
        gp_means, gp_stds = self.predict_with_gp(design_space, train_x, train_y)
        self._last_gp_means = gp_means
        self._last_gp_stds = gp_stds

        # ----------------------------------------------------------------
        # Step 2: Per-principle scoring → overall action type
        # ----------------------------------------------------------------
        action_info = self.principle_scorer.score_principles(self.principle_buffer)
        self._last_action_info = action_info
        action_type = action_info.get("action_type", "explore")

        logger.info("PiGollum action: %s", action_type.upper())
        print(f"\n{'='*60}")
        print(f"[PiGollum] Action: {action_type.upper()}")
        print(f"[PiGollum] {action_info['suggestion']}")
        print(f"{'='*60}\n")

        # ----------------------------------------------------------------
        # Step 3: Acquisition scores (GP-based, driven by action type)
        # ----------------------------------------------------------------
        gp_mean_flat = gp_means.mean(axis=1) if gp_means.ndim > 1 else gp_means
        gp_std_flat  = gp_stds.mean(axis=1)  if gp_stds.ndim  > 1 else gp_stds

        if action_type == "refine":
            acq_scores = self._normalize_scores(gp_mean_flat)
        elif action_type == "validate":
            acq_scores = self._normalize_scores(gp_mean_flat + 0.5 * gp_std_flat)
        else:  # explore
            acq_scores = self._normalize_scores(gp_std_flat)

        # Blend with principle alignment scores when enough principles exist
        use_principles = self.principle_buffer.size >= self.min_principles_for_guidance
        if use_principles:
            candidate_descs = [describe_sequence(s) for s in candidate_sequences]
            principle_scores = self.principle_scorer.score_candidates(
                candidate_descriptions=candidate_descs,
                buffer=self.principle_buffer,
            )
            p_weight = self._get_principle_weight()
            final_scores = (1.0 - p_weight) * acq_scores + p_weight * principle_scores
        else:
            principle_scores = np.zeros(len(candidate_sequences), dtype=np.float32)
            final_scores = acq_scores

        logger.info(
            "PiGollumOptimizer: acq_mean=%.4f  principle_mean=%.4f  final_mean=%.4f",
            acq_scores.mean(), principle_scores.mean(), final_scores.mean(),
        )

        # ----------------------------------------------------------------
        # Step 4: 3-agent pipeline
        #   Agent 1 (Planner)   – re-ranks principles using (pk, yk)
        #   Agent 2 (Hypothesis) – directional hypothesis from action-annotated principles
        #   Agent 3 (Scorer)    – re-ranks top-k by hypothesis alignment
        # ----------------------------------------------------------------
        if self.enable_post_acq_agents and self.principle_extractor is not None \
                and self.principle_buffer.size > 0:

            # Agent 1: Planner
            logger.info("PiGollum [Agent 1 / Planner]: re-ranking principles with (pk, yk)…")
            ranked_principles, actions = self.principle_extractor.rerank_principles(
                buffer=self.principle_buffer,
                action_info=action_info,
                oracle_principle=self._last_oracle_principle,
                oracle_outcome=self._last_oracle_y,
            )
            self._last_planner_response = self.principle_extractor.last_rerank_response

            # Agent 2: Hypothesis
            logger.info("PiGollum [Agent 2 / Hypothesis]: generating directional hypothesis…")
            exp_data = self._get_oracle_history() if self.include_experimental_data else None
            direction_hyp = self.principle_extractor.generate_directional_hypothesis(
                ranked_principles=ranked_principles,
                actions=actions,
                experimental_data=exp_data,
            )
            self._last_direction_hyp = direction_hyp

            # Agent 3: Scorer
            top_k = min(self.top_k_for_rescoring, len(final_scores))
            top_k_idx = np.argsort(final_scores)[-top_k:]
            logger.info(
                "PiGollum [Agent 3 / Scorer]: scoring top-%d candidates by hypothesis…", top_k
            )
            top_k_seqs = [candidate_sequences[i] for i in top_k_idx]
            rescored = self.principle_extractor.score_candidates_by_hypothesis(
                candidates=top_k_seqs,
                hypothesis=direction_hyp,
                gp_means=gp_means[top_k_idx],
                gp_stds=gp_stds[top_k_idx],
                acq_scores=final_scores[top_k_idx],
            )
            self._last_scorer_response = self.principle_extractor.last_scorer_response

            # Redistribute original top-k score magnitudes according to Scorer ranking
            original_top_k = final_scores[top_k_idx].copy()
            sorted_desc = np.sort(original_top_k)[::-1]
            new_rank = np.argsort(rescored)[::-1]
            new_top_k = np.empty(top_k)
            for rank, local_i in enumerate(new_rank):
                new_top_k[local_i] = sorted_desc[rank]
            final_scores = final_scores.copy()
            final_scores[top_k_idx] = new_top_k

        # ----------------------------------------------------------------
        # Step 5: Greedy batch selection
        # ----------------------------------------------------------------
        featurizer = getattr(self.surrogate_model, "finetuning_model", None)
        if featurizer is None:
            z_baseline = train_x
            z_choices = design_space
        else:
            featurizer = featurizer.eval()
            with torch.no_grad():
                z_baseline = featurizer(train_x).to(**self.tkwargs)
                z_choices = featurizer(design_space).to(**self.tkwargs)

        return self._greedy_select(
            final_scores=final_scores,
            design_space=design_space,
            z_baseline=z_baseline,
            z_choices=z_choices,
        )

    # ------------------------------------------------------------------
    # Principle update (called after real oracle evaluation)
    # ------------------------------------------------------------------

    def update_principles(
        self,
        sequence: str,
        outcome: Dict[str, float],
        iteration: Optional[int] = None,
        source: str = "oracle",
    ) -> Optional[Principle]:
        """
        After an oracle evaluation: extract a principle, update (pk, yk) state,
        and add the principle to the buffer.

        pk = the existing principle most similar to the oracle sequence (by cosine
             similarity of sentence embeddings). Stored for the Planner next iteration.
        yk = the oracle outcome dict.
        """
        if self.principle_extractor is None:
            logger.debug("No extractor configured, skipping principle update.")
            return None

        iter_idx = iteration if iteration is not None else self._iteration
        self._iteration = iter_idx + 1

        # ---- Find pk: existing principle closest to the oracle sequence ----
        if self.principle_buffer.size > 0:
            seq_desc = describe_sequence(sequence)
            seq_emb = self.principle_scorer.embed_single(seq_desc)
            if seq_emb is not None:
                existing = self.principle_buffer.get_all()
                emb_list = [p.embedding for p in existing if p.embedding is not None]
                prin_list = [p for p in existing if p.embedding is not None]
                if emb_list:
                    emb_matrix = np.stack(emb_list)
                    sims = PrincipleScorer._cosine_similarity_cross(
                        seq_emb[None], emb_matrix
                    )[0]
                    self._last_oracle_principle = prin_list[int(np.argmax(sims))]

        self._last_oracle_y = dict(outcome)

        # ---- Extract principle from (sequence, outcome) ----
        principle = self.principle_extractor.extract(
            sequence=sequence,
            outcome=outcome,
            iteration=iter_idx,
        )

        if principle is None:
            return None

        principle.source = source

        emb = self.principle_scorer.embed_single(principle.principle_text)
        if emb is not None:
            principle.embedding = emb

        self.principle_buffer.add(principle)

        self._oracle_history.append({
            "sequence_desc": describe_sequence(sequence),
            "outcome": dict(outcome),
        })

        return principle

    # ------------------------------------------------------------------
    # Warm-start: broad principles → evidence-based refinement
    # ------------------------------------------------------------------

    def warm_start_with_refinement(
        self,
        train_sequences: List[str],
        train_outcomes: List[Dict[str, float]],
        n_broad_principles: int = 5,
    ) -> None:
        """
        PiFlow-style warm-start:
        1. Generate broad domain-knowledge principles
        2. For each broad principle, find relevant training examples
        3. Refine each principle with supporting/contradicting evidence

        Parameters
        ----------
        train_sequences : List[str]
            Initial training sequences.
        train_outcomes : List[Dict[str, float]]
            Corresponding outcomes for each training sequence.
        n_broad_principles : int
            Number of broad principles to generate.
        """
        if self.principle_extractor is None:
            logger.warning("No extractor configured; skipping warm-start.")
            return

        # Step 1: Generate broad principles
        logger.info("Warm-start Phase 1: Generating %d broad principles…", n_broad_principles)
        broad_principles = self.principle_extractor.generate_broad_principles(n_broad_principles)

        for i, broad_text in enumerate(broad_principles):
            logger.info("  Broad principle %d: %s", i + 1, broad_text[:100])

            # Add broad principle to buffer first
            broad_principle = Principle(
                sequence="",
                outcome={},
                primary_reward=0.0,
                hypothesis="(domain-knowledge seed principle)",
                principle_text=broad_text,
                iteration=-1,
                source="broad",
            )
            emb = self.principle_scorer.embed_single(broad_text)
            if emb is not None:
                broad_principle.embedding = emb
            self.principle_buffer.add(broad_principle)

        # Step 2: For each broad principle, find relevant training examples
        # and refine with evidence
        logger.info("Warm-start Phase 2: Refining principles with training evidence…")

        # Embed all broad principles
        broad_texts = broad_principles
        broad_embs = self.principle_scorer.embed(broad_texts)

        # Embed all training sequence descriptions
        train_descs = [describe_sequence(seq) for seq in train_sequences]
        train_embs = self.principle_scorer.embed(train_descs)

        if broad_embs is None or train_embs is None:
            logger.warning("Embedding failed; skipping refinement phase.")
            return

        # Cross-similarity: broad principles vs training descriptions
        sim_matrix = self.principle_scorer._cosine_similarity_cross(
            broad_embs, train_embs
        )  # (n_broad, n_train)

        # Get median outcome for support/contradict classification
        primary_key = self.principle_extractor.objective_names[0]
        primary_rewards = [out.get(primary_key, 0.0) for out in train_outcomes]
        median_reward = float(np.median(primary_rewards))

        for i, broad_text in enumerate(broad_texts):
            # Find most similar training sequences
            similarities = sim_matrix[i]  # (n_train,)
            top_indices = np.argsort(similarities)[-10:][::-1]  # top-10 most relevant

            supporting = []
            contradicting = []

            for idx in top_indices:
                evidence = {
                    "sequence_desc": train_descs[idx],
                    "outcome": train_outcomes[idx],
                }
                reward = train_outcomes[idx].get(primary_key, 0.0)
                if reward >= median_reward:
                    supporting.append(evidence)
                else:
                    contradicting.append(evidence)

            # Refine the broad principle with evidence
            try:
                refined_text = self.principle_extractor.refine_principle_with_evidence(
                    principle_text=broad_text,
                    supporting_evidence=supporting,
                    contradicting_evidence=contradicting,
                )
            except Exception as e:
                logger.warning("Refinement failed for principle %d: %s", i + 1, e)
                continue

            # Compute average reward of supporting evidence as proxy reward
            support_rewards = [ev["outcome"].get(primary_key, 0.0) for ev in supporting]
            avg_reward = float(np.mean(support_rewards)) if support_rewards else 0.0

            refined_principle = Principle(
                sequence="",
                outcome={},
                primary_reward=avg_reward,
                hypothesis=f"Refined from broad principle: {broad_text[:100]}",
                principle_text=refined_text,
                iteration=-1,
                source="refined",
            )
            emb = self.principle_scorer.embed_single(refined_text)
            if emb is not None:
                refined_principle.embedding = emb
            self.principle_buffer.add(refined_principle)

            logger.info(
                "  Refined principle %d (avg_reward=%.4f): %s",
                i + 1, avg_reward, refined_text[:100],
            )

        logger.info(
            "Warm-start complete: %d principles in buffer (%d broad + %d refined)",
            self.principle_buffer.size,
            len(broad_principles),
            self.principle_buffer.size - len(broad_principles),
        )

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _get_principle_weight(self) -> float:
        """Return the current principle weight (possibly scheduled)."""
        if self.principle_weight_schedule is None or self.principle_weight_schedule == "constant":
            return self.principle_weight

        n = self.principle_buffer.size
        max_alpha = self.principle_weight

        if self.principle_weight_schedule == "linear":
            return min(max_alpha, max_alpha * n / 10.0)

        if self.principle_weight_schedule == "step":
            steps = n // 2
            return min(max_alpha, steps * 0.05)

        return self.principle_weight

    def _get_oracle_history(self) -> List[Dict]:
        """Return the list of oracle evaluations accumulated so far."""
        return list(self._oracle_history)

    @staticmethod
    def _normalize_scores(scores: np.ndarray) -> np.ndarray:
        lo, hi = scores.min(), scores.max()
        if hi - lo < 1e-8:
            return np.ones_like(scores) * 0.5
        return (scores - lo) / (hi - lo)

    def _greedy_select(
        self,
        final_scores: np.ndarray,
        design_space: Tensor,
        z_baseline: Tensor,
        z_choices: Tensor,
    ) -> List[Tensor]:
        """Greedy batch selection on pre-computed final_scores."""
        M = design_space.size(0)
        choices_idx = torch.arange(M, device=self.tkwargs["device"])
        final_scores_t = torch.tensor(final_scores, device=self.tkwargs["device"])

        candidates = []
        selected_indices = []
        pending = torch.empty(0, z_choices.size(-1), **self.tkwargs)
        eval_scores = final_scores_t.clone()
        eval_idx = choices_idx.clone()

        steps = min(self.batch_size, M)

        for step in range(steps):
            best_local = int(eval_scores.argmax().item())
            global_idx = int(eval_idx[best_local].item())

            candidates.append(design_space[global_idx].reshape(-1))
            selected_indices.append(global_idx)

            keep = torch.ones(eval_scores.size(0), dtype=torch.bool)
            keep[best_local] = False
            eval_scores = eval_scores[keep]
            eval_idx = eval_idx[keep]

            best_z = z_choices[global_idx]
            pending = torch.cat([pending, best_z.unsqueeze(0)], dim=0)

        self._last_selected_indices = selected_indices
        return candidates
