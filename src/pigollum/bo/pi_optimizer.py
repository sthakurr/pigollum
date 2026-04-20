"""
PiGollumOptimizer: Principle-guided Bayesian Optimization with GP as
experiment agent.

Redesigned to follow PiFlow's architecture where the GP acts as an oracle
(experiment agent) that validates LLM-generated hypotheses, and the feedback
loop between hypothesis generation and GP validation refines principles.

Architecture
────────────
    Warm-start (once)
    ─────────────────
    1. LLM generates broad domain-knowledge principles (no data)
    2. For each broad principle, retrieve relevant training sequences
    3. LLM refines each principle with supporting/contradicting evidence
    → Result: grounded, task-level seed principles

    Per BO iteration
    ────────────────
    1. Train GP surrogate on (train_x, train_y)
    2. GP predicts mean + std for all candidates
    3. [Inner PiFlow loop — n_inner_steps]:
       a. Planner: analyze principles → suggest action (explore/refine/validate)
       b. Planner selects candidates based on action + GP uncertainty
       c. LLM (Hypothesis Agent): predict sequence performance, grounded in principles
       d. GP (Experiment Agent): provide prediction as validation
       e. Extract new principle from (hypothesis, GP prediction)
    4. After inner loop: combine acquisition scores + principle scores
    5. Select best candidate(s) for real oracle evaluation
    6. Oracle evaluation → extract principle from real data

The ``principle_weight`` (α) can be scheduled to increase as more principles
accumulate, gradually shifting trust from pure BO toward principle guidance.
"""
import logging
import sys
import os
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from torch import Tensor

# Add gollum to path (it lives in pigollum/gollum-2)
_GOLLUM_SRC = os.path.join(os.path.dirname(__file__), "..", "..", "..", "..", "gollum-2", "src")
if os.path.isdir(_GOLLUM_SRC):
    sys.path.insert(0, os.path.abspath(_GOLLUM_SRC))

from gollum.bo.optimizer import BotorchOptimizer

from pigollum.principle.buffer import Principle, PrincipleBuffer
from pigollum.principle.extractor import PrincipleExtractor
from pigollum.principle.scorer import PrincipleScorer
from pigollum.principle.planner import Planner
from pigollum.principle.journal import PrincipleJournal
from pigollum.utils.sequence_utils import describe_sequence

logger = logging.getLogger(__name__)


class PiGollumOptimizer(BotorchOptimizer):
    """
    Principle-guided BO optimizer with GP-as-experiment-agent inner loop.

    Parameters
    ----------
    principle_weight : float
        α ∈ [0, 1].  Weight of principle scores relative to BO acquisition scores.
    min_principles_for_guidance : int
        Minimum number of principles before principle guidance activates.
    principle_weight_schedule : str | None
        Schedule for α: None/"constant", "linear", or "step".
    n_inner_steps : int
        Number of hypothesis-validation cycles in the inner PiFlow loop.
    candidate_sample_size : int
        Number of candidates to consider per inner step.
    extractor : PrincipleExtractor | None
        LLM-based hypothesis and principle extraction.
    scorer : PrincipleScorer | None
        Exploration-exploitation scorer.
    planner : Planner | None
        Inner loop orchestrator.
    buffer : PrincipleBuffer | None
        Pre-populated buffer.
    All other kwargs are forwarded to BotorchOptimizer.
    """

    def __init__(
        self,
        *args,
        principle_weight: float = 0.3,
        min_principles_for_guidance: int = 3,
        principle_weight_schedule: Optional[str] = None,
        n_inner_steps: int = 5,
        candidate_sample_size: int = 10,
        extractor: Optional[PrincipleExtractor] = None,
        scorer: Optional[PrincipleScorer] = None,
        planner: Optional[Planner] = None,
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
        self.n_inner_steps = n_inner_steps
        self.enable_post_acq_agents = enable_post_acq_agents
        self.top_k_for_rescoring = top_k_for_rescoring
        self.include_experimental_data = include_experimental_data

        self.principle_buffer = buffer or PrincipleBuffer()
        self.principle_extractor = extractor  # may be None
        self.principle_scorer = scorer or PrincipleScorer(device=str(self.tkwargs["device"]))
        self.planner = planner or Planner(
            n_inner_steps=n_inner_steps,
            candidate_sample_size=candidate_sample_size,
        )
        self.journal: Optional[PrincipleJournal] = None

        self._iteration = 0
        self._last_action_info: dict = {}
        self._last_gp_means: Optional[np.ndarray] = None
        self._last_gp_stds: Optional[np.ndarray] = None
        self._iteration_gp_data: List[Dict] = []   # accumulates per inner loop step
        self._oracle_history: List[Dict] = []       # accumulates across all iterations

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
        self.surrogate_model.model.to(**self.tkwargs)
        self.surrogate_model.model.eval()

        featurizer = getattr(self.surrogate_model, "finetuning_model", None)
        if featurizer is not None:
            featurizer = featurizer.eval()
            with torch.no_grad():
                z = featurizer(design_space.to(**self.tkwargs)).to(**self.tkwargs)
        else:
            z = design_space.to(**self.tkwargs)

        # Get posterior predictions
        with torch.no_grad():
            posterior = self.surrogate_model.model.posterior(z)
            means = posterior.mean.cpu().numpy()       # (M, n_obj)
            stds = posterior.variance.sqrt().cpu().numpy()  # (M, n_obj)

        return means, stds

    # ------------------------------------------------------------------
    # Inner PiFlow loop (GP as experiment agent)
    # ------------------------------------------------------------------

    def run_inner_piflow_loop(
        self,
        candidate_sequences: List[str],
        gp_means: np.ndarray,
        gp_stds: np.ndarray,
        objective_names: List[str],
    ) -> None:
        """
        Run the inner PiFlow loop: hypothesis generation → GP validation →
        principle extraction, repeated for n_inner_steps.

        Parameters
        ----------
        candidate_sequences : List[str]
            All candidate sequences in the design space.
        gp_means, gp_stds : np.ndarray
            GP predictions for all candidates. Shape (M, n_obj).
        objective_names : List[str]
            Names of objectives for outcome dict construction.
        """
        if self.principle_extractor is None:
            logger.debug("No extractor configured, skipping inner PiFlow loop.")
            return

        for step in range(self.n_inner_steps):
            logger.info("  Inner PiFlow step %d / %d", step + 1, self.n_inner_steps)

            # 1. Scorer suggests action based on current principles
            action_info = self.principle_scorer.score_principles(self.principle_buffer)
            self._last_action_info = action_info

            # 2. Planner generates guidance and selects candidates
            guidance, selected_indices = self.planner.plan_next_hypothesis(
                action_info=action_info,
                buffer=self.principle_buffer,
                candidate_sequences=candidate_sequences,
                gp_means=gp_means,
                gp_stds=gp_stds,
            )

            # 3. For each selected candidate: hypothesis → GP validation → principle
            for cand_idx in selected_indices[:3]:  # limit per step to avoid LLM overload
                seq = candidate_sequences[cand_idx]

                # LLM generates predictive hypothesis
                try:
                    hypothesis = self.principle_extractor.generate_prediction_hypothesis(
                        sequence=seq,
                        guidance=guidance,
                    )
                except Exception as e:
                    logger.warning("Hypothesis generation failed: %s", e)
                    continue

                # GP (experiment agent) provides prediction
                gp_mean = gp_means[cand_idx]  # (n_obj,)
                gp_std = gp_stds[cand_idx]    # (n_obj,)

                # Build outcome dict from GP prediction
                gp_outcome = {}
                for j, name in enumerate(objective_names):
                    gp_outcome[name] = float(gp_mean[j]) if gp_mean.ndim > 0 else float(gp_mean)

                # GP confidence: higher when std is lower
                avg_std = float(gp_std.mean()) if gp_std.ndim > 0 else float(gp_std)
                gp_confidence = 1.0 / (1.0 + avg_std)

                # Extract principle from (hypothesis, GP prediction)
                try:
                    principle_text = self.principle_extractor._extract_principle(
                        hypothesis=hypothesis,
                        outcome=gp_outcome,
                    )
                except Exception as e:
                    logger.warning("Principle extraction failed: %s", e)
                    continue

                primary_key = objective_names[0]
                primary_reward = gp_outcome.get(primary_key, 0.0)

                principle = Principle(
                    sequence=seq,
                    outcome=gp_outcome,
                    primary_reward=primary_reward,
                    hypothesis=hypothesis,
                    principle_text=principle_text,
                    iteration=self._iteration,
                    source="gp",
                    gp_confidence=gp_confidence,
                )

                # Embed and store
                emb = self.principle_scorer.embed_single(principle.principle_text)
                if emb is not None:
                    principle.embedding = emb
                self.principle_buffer.add(principle)

                # Track for post-acquisition agents
                self._iteration_gp_data.append({
                    "hypothesis": hypothesis,
                    "outcome": gp_outcome,
                })

            logger.info(
                "  Inner step %d complete. Buffer size: %d",
                step + 1, self.principle_buffer.size,
            )

    # ------------------------------------------------------------------
    # Core BO loop with inner PiFlow loop
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
        PiFlow-style BO loop:
        1. Train GP surrogate
        2. Get GP predictions (means + stds) for all candidates
        3. Run inner PiFlow loop (hypothesis → GP validation → principle extraction)
        4. Principle-driven selection: GP predictions + principle alignment
           (NO acquisition function blending — the GP's role is as experiment
           agent, not as a traditional BO acquisition function)

        Falls back to standard BO only if no sequences or extractor are provided.
        """
        has_sequences = (
            candidate_sequences is not None
            and len(candidate_sequences) == design_space.size(0)
        )

        # Always train the GP first
        train_x = train_x.to(**self.tkwargs)
        train_y = train_y.to(**self.tkwargs)
        design_space = design_space.to(**self.tkwargs)

        if not test:
            self.train_surrogate_model(train_x, train_y)

        # If no sequences or no extractor, fall back to pure BO
        if not has_sequences or self.principle_extractor is None:
            logger.info("PiGollumOptimizer: using pure BO (no sequences or extractor).")
            return super().suggest_next_experiments(train_x, train_y, design_space, test=True)

        # ----------------------------------------------------------------
        # Step 1: Get GP predictions for all candidates (experiment agent)
        # ----------------------------------------------------------------
        logger.info("PiGollumOptimizer: GP predicting all candidates…")
        gp_means, gp_stds = self.predict_with_gp(design_space, train_x, train_y)
        self._last_gp_means = gp_means
        self._last_gp_stds = gp_stds

        # ----------------------------------------------------------------
        # Step 2: Run inner PiFlow loop (hypothesis → GP validation → principles)
        # ----------------------------------------------------------------
        self._iteration_gp_data = []  # reset for this iteration

        if objective_names is not None:
            logger.info("PiGollumOptimizer: running inner PiFlow loop (%d steps)…", self.n_inner_steps)
            self.run_inner_piflow_loop(
                candidate_sequences=candidate_sequences,
                gp_means=gp_means,
                gp_stds=gp_stds,
                objective_names=objective_names,
            )

        # ----------------------------------------------------------------
        # Step 3: Principle-driven selection (PiFlow-style)
        # ----------------------------------------------------------------
        # In PiFlow, there is no acquisition function blending.  The inner
        # loop refines principles, then the Planner uses those principles
        # + GP predictions to select the final candidates.  This replaces
        # the old (1-α)*acq + α*principle weighted combination.
        # ----------------------------------------------------------------

        action_info = self.principle_scorer.score_principles(self.principle_buffer)
        self._last_action_info = action_info
        action_type = action_info.get("action_type", "explore")

        logger.info("PiGollum action: %s", action_type.upper())
        print(f"\n{'='*60}")
        print(f"[PiGollum] Action: {action_type.upper()}")
        print(f"[PiGollum] {action_info['suggestion']}")
        print(f"{'='*60}\n")

        use_principles = self.principle_buffer.size >= self.min_principles_for_guidance

        if use_principles:
            # Principle alignment scores
            candidate_descs = [describe_sequence(s) for s in candidate_sequences]
            principle_scores = self.principle_scorer.score_candidates(
                candidate_descriptions=candidate_descs,
                buffer=self.principle_buffer,
            )  # (M,) ∈ [0, 1]
        else:
            principle_scores = np.zeros(len(candidate_sequences), dtype=np.float32)

        # GP-derived scores based on the Planner's action
        gp_mean_flat = gp_means.mean(axis=1) if gp_means.ndim > 1 else gp_means
        gp_std_flat = gp_stds.mean(axis=1) if gp_stds.ndim > 1 else gp_stds

        if action_type == "refine":
            # Exploit: high GP mean + principle alignment
            gp_scores = self._normalize_scores(gp_mean_flat)
        elif action_type == "validate":
            # Upper confidence bound: balance mean + uncertainty
            gp_scores = self._normalize_scores(gp_mean_flat + 0.5 * gp_std_flat)
        else:  # explore
            # Explore: prioritise GP uncertainty (regions GP knows least)
            gp_scores = self._normalize_scores(gp_std_flat)

        # Final score: GP (experiment agent view) + principle alignment
        # GP provides the "what is promising/uncertain" signal.
        # Principles provide the "what is mechanistically sound" signal.
        if use_principles:
            # Weight principles more as they accumulate and gain confidence
            p_weight = self._get_principle_weight()
            final_scores = (1.0 - p_weight) * gp_scores + p_weight * principle_scores
        else:
            final_scores = gp_scores

        logger.info(
            "PiGollumOptimizer: gp_mean=%.4f  principle_mean=%.4f  final_mean=%.4f",
            gp_scores.mean(), principle_scores.mean(), final_scores.mean(),
        )

        # ----------------------------------------------------------------
        # Step 3.5: Post-acquisition 3-agent pipeline
        # Runs AFTER acquisition scores are computed, BEFORE greedy select.
        # Agent 1 (Planner)  – re-ranks principles given new iteration data
        # Agent 2 (Hypothesis) – generates directional hypothesis for next step
        # Agent 3 (Scorer)   – re-ranks top-k candidates by hypothesis alignment
        # ----------------------------------------------------------------
        if self.enable_post_acq_agents and self.principle_extractor is not None:
            top_k = min(self.top_k_for_rescoring, len(final_scores))
            top_k_idx = np.argsort(final_scores)[-top_k:]

            # Agent 1: Planner re-ranks principles
            logger.info("PiGollum [Agent 1]: re-ranking principles…")
            best_so_far: Dict = {}
            if self.journal is not None and hasattr(self.journal, "_iterations") \
                    and self.journal._iterations:
                last = self.journal._iterations[-1]
                best_so_far = last.get("best_so_far", {})
            ranked_principles = self.principle_extractor.rerank_principles(
                buffer=self.principle_buffer,
                iteration_data={
                    "gp_hypotheses": self._iteration_gp_data,
                    "best_so_far": best_so_far,
                },
            )

            # Agent 2: Hypothesis generates directional hypothesis
            logger.info("PiGollum [Agent 2]: generating directional hypothesis…")
            exp_data = (
                self._get_oracle_history() if self.include_experimental_data else None
            )
            direction_hyp = self.principle_extractor.generate_directional_hypothesis(
                ranked_principles=ranked_principles,
                experimental_data=exp_data,
            )
            logger.info(
                "PiGollum directional hypothesis: %s…",
                direction_hyp[:120].replace("\n", " "),
            )
            print(f"\n[PiGollum] Directional Hypothesis:\n{direction_hyp}\n")

            # Agent 3: Scorer re-ranks top-k candidates by hypothesis alignment
            logger.info(
                "PiGollum [Agent 3]: scoring top-%d candidates by hypothesis…", top_k
            )
            top_k_seqs = [candidate_sequences[i] for i in top_k_idx]
            rescored = self.principle_extractor.score_candidates_by_hypothesis(
                candidates=top_k_seqs,
                hypothesis=direction_hyp,
                gp_means=gp_means[top_k_idx],
                gp_stds=gp_stds[top_k_idx],
                acq_scores=final_scores[top_k_idx],
            )

            # Redistribute original top-k score magnitudes according to new ranking,
            # preserving the top-k's advantage over the rest of the pool.
            original_top_k = final_scores[top_k_idx].copy()
            sorted_desc = np.sort(original_top_k)[::-1]
            new_rank = np.argsort(rescored)[::-1]  # local indices, best first
            new_top_k = np.empty(top_k)
            for rank, local_i in enumerate(new_rank):
                new_top_k[local_i] = sorted_desc[rank]
            final_scores = final_scores.copy()
            final_scores[top_k_idx] = new_top_k

        # ----------------------------------------------------------------
        # Step 4: Greedy batch selection
        # ----------------------------------------------------------------
        # Compute featurized representations for _greedy_select
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
        After an oracle evaluation, extract a principle and add it to the buffer.

        Parameters
        ----------
        sequence : str
            The evaluated amino acid sequence.
        outcome : Dict[str, float]
            Observed objectives.
        iteration : int, optional
            Current BO iteration index.
        source : str
            "oracle" for real experimental data, "gp" for GP predictions.

        Returns
        -------
        Principle | None
        """
        if self.principle_extractor is None:
            logger.debug("No extractor configured, skipping principle update.")
            return None

        iter_idx = iteration if iteration is not None else self._iteration
        self._iteration = iter_idx + 1

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

        # Track for Agent 2's experimental context
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
