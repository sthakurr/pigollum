"""
Planner: PiFlow-style orchestrator for the inner hypothesis-validation loop.

The Planner sits between the PrincipleScorer (which suggests actions) and the
PrincipleExtractor (which generates hypotheses).  It translates the scorer's
explore/refine/validate signals into concrete guidance for hypothesis generation,
mirroring PiFlow's Planner Agent.

Architecture
────────────
    PrincipleScorer.score_principles(buffer)
        │
        ▼
    Planner.plan_next_hypothesis(action_info, buffer, ...)
        │  ─ synthesises guidance from action_info + best principles
        ▼
    PrincipleExtractor.generate_hypothesis_for_sequence(seq, guidance)
        │
        ▼
    GP (Experiment Agent) predicts performance
        │
        ▼
    PrincipleExtractor.extract_principle(hypothesis, gp_outcome)
        │
        ▼
    PrincipleBuffer.add(new_principle)

The inner loop runs ``n_inner_steps`` times per BO iteration, accumulating
principles that are then used to guide candidate selection.
"""
import logging
from typing import Dict, List, Optional, Tuple

import numpy as np

from pigollum.principle.buffer import PrincipleBuffer

logger = logging.getLogger(__name__)


class Planner:
    """
    Orchestrates the inner PiFlow loop within each BO iteration.

    Responsibilities (mirroring PiFlow's Planner Agent):
    1. Interpret the action suggestion from PrincipleScorer
    2. Select candidate sequences for hypothesis generation
    3. Synthesise guidance that steers the LLM toward the right action
    4. Track the inner loop state

    Parameters
    ----------
    n_inner_steps : int
        Number of hypothesis-validation cycles per BO iteration.
    candidate_sample_size : int
        Number of candidate sequences to consider per inner step.
        Selected by GP uncertainty (explore) or GP mean (refine/validate).
    task_context : str
        Task description injected into guidance prompts.
    """

    def __init__(
        self,
        n_inner_steps: int = 5,
        candidate_sample_size: int = 10,
        task_context: str = "",
    ) -> None:
        self.n_inner_steps = n_inner_steps
        self.candidate_sample_size = candidate_sample_size
        self.task_context = task_context

    def plan_next_hypothesis(
        self,
        action_info: Dict,
        buffer: PrincipleBuffer,
        candidate_sequences: List[str],
        gp_means: Optional[np.ndarray] = None,
        gp_stds: Optional[np.ndarray] = None,
    ) -> Tuple[str, List[int]]:
        """
        Generate guidance for the hypothesis agent and select candidate
        sequences to hypothesise about.

        Parameters
        ----------
        action_info : dict
            Output from PrincipleScorer.score_principles().
        buffer : PrincipleBuffer
            Current principle buffer.
        candidate_sequences : List[str]
            All available candidate sequences.
        gp_means : np.ndarray, optional
            GP mean predictions for each candidate (N,) or (N, n_obj).
        gp_stds : np.ndarray, optional
            GP standard deviation for each candidate.

        Returns
        -------
        guidance : str
            Text guidance for the hypothesis agent (injected into LLM prompt).
        selected_indices : List[int]
            Indices into candidate_sequences to generate hypotheses for.
        """
        action_type = action_info.get("action_type", "explore")
        suggestion = action_info.get("suggestion", "")

        # Select candidate indices based on action type and GP predictions
        selected_indices = self._select_candidates(
            action_type=action_type,
            candidate_sequences=candidate_sequences,
            gp_means=gp_means,
            gp_stds=gp_stds,
        )

        # Build guidance text
        guidance = self._build_guidance(
            action_type=action_type,
            suggestion=suggestion,
            buffer=buffer,
        )

        return guidance, selected_indices

    def _select_candidates(
        self,
        action_type: str,
        candidate_sequences: List[str],
        gp_means: Optional[np.ndarray],
        gp_stds: Optional[np.ndarray],
    ) -> List[int]:
        """
        Select which candidates to generate hypotheses for.

        - explore: pick candidates with highest GP uncertainty
        - refine:  pick candidates with highest GP predicted mean
        - validate: mix of high-mean and moderate-uncertainty candidates
        """
        n = len(candidate_sequences)
        k = min(self.candidate_sample_size, n)

        if gp_means is None or gp_stds is None:
            # No GP predictions yet — random selection
            return np.random.choice(n, size=k, replace=False).tolist()

        # Flatten multi-objective to scalar for selection
        means = gp_means if gp_means.ndim == 1 else gp_means.mean(axis=1)
        stds = gp_stds if gp_stds.ndim == 1 else gp_stds.mean(axis=1)

        if action_type == "explore":
            # Highest uncertainty — regions GP knows least about
            scores = stds
        elif action_type == "refine":
            # Highest predicted mean — exploit known good regions
            scores = means
        elif action_type == "validate":
            # Upper confidence bound — balance mean + uncertainty
            scores = means + 0.5 * stds
        else:
            scores = stds  # default to exploration

        top_indices = np.argsort(scores)[-k:][::-1]
        return top_indices.tolist()

    def _build_guidance(
        self,
        action_type: str,
        suggestion: str,
        buffer: PrincipleBuffer,
    ) -> str:
        """
        Build guidance text for the hypothesis agent, mirroring PiFlow's
        Planner → Hypothesis Agent communication.

        The guidance includes:
        1. The action type and PrincipleScorer's suggestion
        2. The best current principles (for context)
        3. Specific instructions based on the action type
        """
        parts = []

        # Action context
        parts.append(f"ACTION: {action_type.upper()}")
        parts.append(f"PRINCIPLE GUIDANCE: {suggestion}")
        parts.append("")

        # Best principles for context
        if buffer.size > 0:
            best = buffer.best_principle()
            parts.append("CURRENT BEST PRINCIPLE:")
            parts.append(f"  {best.principle_text}")
            parts.append(f"  (reward={best.primary_reward:.4f})")
            parts.append("")

            # Show top-3 principles for broader context
            all_principles = buffer.get_all()
            rewards = buffer.rewards
            top_indices = np.argsort(rewards)[-3:][::-1]
            if len(top_indices) > 1:
                parts.append("TOP PRINCIPLES:")
                for idx in top_indices:
                    p = all_principles[idx]
                    parts.append(f"  - [reward={p.primary_reward:.4f}] {p.principle_text[:150]}")
                parts.append("")

        # Action-specific instructions
        if action_type == "refine":
            parts.append(
                "INSTRUCTION: Generate a hypothesis that builds on and extends the best "
                "principle above. Focus on why the candidate sequence's specific features "
                "align with (or improve upon) the principle. Predict quantitative performance."
            )
        elif action_type == "validate":
            parts.append(
                "INSTRUCTION: Generate a hypothesis that tests the reliability of the best "
                "principle. Look for candidate sequences that partially match the principle "
                "to determine its boundary conditions. Predict performance."
            )
        elif action_type == "explore":
            parts.append(
                "INSTRUCTION: Generate a hypothesis based on a DIFFERENT mechanism than "
                "the current best principle. Look for novel sequence features that might "
                "contribute to performance through an alternative pathway. Predict performance."
            )

        return "\n".join(parts)
