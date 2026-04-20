from __future__ import annotations

import numpy as np
import torch
from dataclasses import dataclass
from typing import List, Optional, Tuple


@dataclass
class IterationRecord:
    """Stores the inputs and observed scores from a single BO iteration."""

    iteration: int
    input_texts: List[str]
    scores: List[float]

    @property
    def best_score(self) -> float:
        return max(self.scores)

    @property
    def best_input(self) -> str:
        return self.input_texts[int(np.argmax(self.scores))]


class ReasoningAgent:
    """
    Augments featurizer inputs with a natural-language summary of the
    optimization trajectory so the embedding model produces context-aware
    representations at each BO iteration.

    Two modes
    ---------
    "programmatic"
        Builds a structured, deterministic summary of the full observed
        trajectory and prepends it to every input string.  No extra model
        is loaded.

    "llm"
        Passes the trajectory to a small instruction-tuned LLM
        (default: microsoft/Phi-3-mini-4k-instruct) which synthesises a
        concise scientific rationale before each input string.  The LLM is
        lazy-loaded on the first call to keep startup cost zero when the
        agent is used in programmatic mode.

    Notes
    -----
    Context augmentation works best with text-embedding featurizers that
    understand natural language (e.g. get_huggingface_embeddings,
    instructor_embeddings).  For specialised biological encoders such as
    ProtT5 that space-separate amino-acid sequences internally, the
    prepended context will be treated as part of the sequence — callers
    should verify the featurizer handles mixed text gracefully or use a
    separate embedding head for the context.
    """

    def __init__(
        self,
        mode: str = "programmatic",
        model_name: str = "microsoft/Phi-3-mini-4k-instruct",
        max_new_tokens: int = 150,
        device: Optional[str] = None,
    ):
        if mode not in ("programmatic", "llm"):
            raise ValueError(f"mode must be 'programmatic' or 'llm', got '{mode}'")

        self.mode = mode
        self.model_name = model_name
        self.max_new_tokens = max_new_tokens
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        self.history: List[IterationRecord] = []

        self.last_reasoning: Optional[str] = None  # For LLM mode: store last iteration's rationale

        self._llm = None
        self._tokenizer = None

        print(f"[ReasoningAgent] initialised  mode={mode}  device={self.device}"
              + (f"  llm={model_name}" if mode == "llm" else ""))

    # ------------------------------------------------------------------
    # History management
    # ------------------------------------------------------------------

    def record_observation(
        self,
        iteration: int,
        input_texts: List[str],
        scores: List[float],
    ) -> None:
        """Append the results of one completed BO iteration to the history."""
        record = IterationRecord(
            iteration=iteration,
            input_texts=list(input_texts),
            scores=list(scores),
        )
        self.history.append(record)
        _, best_score = self.best_so_far
        print(
            f"[ReasoningAgent] iteration {iteration} recorded  "
            f"n={len(scores)}  best_this_iter={record.best_score:.4f}  "
            f"global_best={best_score:.4f}"
        )

    @property
    def best_so_far(self) -> Optional[Tuple[str, float]]:
        """Return (best_input, best_score) across the full trajectory, or None."""
        if not self.history:
            return None
        all_pairs = [
            (rec.input_texts[i], rec.scores[i])
            for rec in self.history
            for i in range(len(rec.scores))
        ]
        return max(all_pairs, key=lambda p: p[1])

    # ------------------------------------------------------------------
    # Context prompt construction
    # ------------------------------------------------------------------

    def _programmatic_summary(self) -> str:
        """
        Structured summary of the full trajectory.

        Format
        ------
        Optimization context (N iterations, M experiments observed):
          Iteration 0: best_score=0.8900, best_input=<first 60 chars>
          Iteration 1: best_score=0.9200, best_input=<first 60 chars>
          ...
        Global best: score=0.9200, input=<first 60 chars>
        Evaluate the following:
        """
        total_experiments = sum(len(r.scores) for r in self.history)
        best_input, best_score = self.best_so_far

        lines = [
            f"Optimization context ({len(self.history)} iterations, "
            f"{total_experiments} experiments observed):"
        ]
        for rec in self.history:
            lines.append(
                f"  Iteration {rec.iteration}: "
                f"best_score={rec.best_score:.4f}, "
                f"best_input={rec.best_input[:60]}"
            )
        lines.append(
            f"Global best: score={best_score:.4f}, input={best_input[:60]}"
        )
        lines.append("Evaluate the following:")
        summary = "\n".join(lines)
        print(f"[ReasoningAgent] programmatic context built  "
              f"({len(self.history)} iters, {total_experiments} experiments)")
        return summary

    def _llm_summary(self) -> str:
        """
        LLM-generated scientific rationale for the current search state.

        The LLM receives the full trajectory and is asked to reason about
        (1) patterns in high-scoring inputs, (2) promising search regions,
        and (3) what the next candidate should look like.  Its response is
        used as the context prefix for all inputs passed to the featurizer.
        """
        self._ensure_llm_loaded()

        history_lines: List[str] = []
        for rec in self.history:
            for inp, score in zip(rec.input_texts, rec.scores):
                history_lines.append(f"  - score={score:.4f}  input={inp}")

        history_text = "\n".join(history_lines)
        best_input, best_score = self.best_so_far

        print(f"[ReasoningAgent] history_text: {history_text}...")

        user_content = (
            "You are a scientific advisor guiding a Bayesian optimization "
            "experiment over a discrete chemical search space.\n\n"

            f"Optimization history ({len(self.history)} iterations, "
            f"{sum(len(r.scores) for r in self.history)} experiments):\n"
            f"{history_text}\n\n"

            f"Global best so far: score={best_score:.4f}, input={best_input}\n\n"

            f"Your reasoning from the previous iteration: {self.last_reasoning if self.last_reasoning else 'None yet.'}\n\n"

            "Analyze the experimental history and respond in exactly 3 sentences:\n"
            "(1) CONTRAST: What specific parameters (ligand, base, solvent, aryl halide) "
            "do the TOP 3 experiments share that the BOTTOM 3 do NOT? Name the actual values.\n"
            "(2) GAPS: What parameter combinations have NOT been tried yet that seem worth exploring "
            "given the pattern above?\n"
            "(3) UPDATE: If you gave reasoning in the previous iteration, does the new result "
            "confirm or contradict it? State specifically what changed in your hypothesis.\n"
        )

        messages = [{"role": "user", "content": user_content}]
        prompt = self._tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = self._tokenizer(prompt, return_tensors="pt").to(self.device)
        with torch.no_grad():
            output_ids = self._llm.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                do_sample=False,
                use_cache=False,
            )

        generated = self._tokenizer.decode(
            output_ids[0][inputs.input_ids.shape[1]:],
            skip_special_tokens=True,
        ).strip()
        print(f"[ReasoningAgent] LLM context generated:\n  {generated}")

        self.last_reasoning = generated  # Store for next iteration's UPDATE

        return generated + "\nEvaluate the following:"

    def build_context_prompt(self) -> Optional[str]:
        """
        Return the context prompt for the current iteration, or None if no
        observations have been recorded yet (first iteration).
        """
        if not self.history:
            return None
        if self.mode == "programmatic":
            return self._programmatic_summary()
        return self._llm_summary()

    # ------------------------------------------------------------------
    # Input augmentation and re-featurization
    # ------------------------------------------------------------------

    def augment_inputs(self, input_texts: List[str]) -> List[str]:
        """
        Prepend the current context prompt to every input string.

        Returns the original list unchanged if no history exists yet.
        """
        prompt = self.build_context_prompt()
        if prompt is None:
            return input_texts
        return [f"{prompt}\n{inp}" for inp in input_texts]

    def refeaturize(self, input_texts: List[str], featurizer) -> np.ndarray:
        """
        Augment inputs with the current context prompt then run featurization.

        Parameters
        ----------
        input_texts : List[str]
            Raw input strings (e.g. protein sequences, SMILES, procedures).
        featurizer : gollum.featurization.base.Featurizer
            The configured featurizer instance from the data module.

        Returns
        -------
        np.ndarray of shape (N, D)
            Context-augmented feature matrix.
        """
        print(f"[ReasoningAgent] re-featurizing {len(input_texts)} inputs  mode={self.mode}")
        augmented = self.augment_inputs(input_texts)
        features = featurizer.featurize(augmented)
        print(f"[ReasoningAgent] re-featurization done  shape={features.shape}")
        return features

    # ------------------------------------------------------------------
    # LLM loading (lazy)
    # ------------------------------------------------------------------

    def _ensure_llm_loaded(self) -> None:
        if self._llm is not None:
            return

        from transformers import AutoModelForCausalLM, AutoTokenizer

        print(f"Loading reasoning LLM: {self.model_name}")
        self._tokenizer = AutoTokenizer.from_pretrained(
            self.model_name, trust_remote_code=True
        )
        self._llm = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            torch_dtype=torch.bfloat16,
            device_map=self.device,
            trust_remote_code=True,
        )
        self._llm.eval()
        print(f"Reasoning LLM loaded on {self.device}")
