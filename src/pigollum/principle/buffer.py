"""
PrincipleBuffer: Stores and manages discovered scientific principles.

Each Principle captures a (hypothesis, experiment, extracted_principle) triple,
mirroring PiFlow's Principle dataclass but adapted for gollum's BO context where
inputs are amino acid sequences and outputs are multi-objective targets.
"""
import uuid
import json
import logging
from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List, Optional
import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class Principle:
    """
    A discovered scientific principle linking a sequence, its observed outcomes,
    a hypothesis about why it performed as it did, and an extracted principle.

    Attributes:
        id:              Unique identifier (UUID4 string).
        sequence:        Raw amino acid sequence string that was evaluated.
                         May be empty for broad/refined principles not tied to
                         a single sequence.
        outcome:         Dict mapping objective name -> observed value
                         (e.g. {"yield": 62.0, "enantioselectivity": 0.04}).
                         For GP-validated principles, these are GP predictions.
        primary_reward:  Scalar reward used for exploration-exploitation scoring.
                         By convention, the first maximisation objective.
        hypothesis:      LLM-generated hypothesis explaining the outcome in terms
                         of sequence-level biochemical features.
        principle_text:  LLM-extracted concise scientific principle (1-2 sentences).
        embedding:       Sentence embedding of principle_text (np.ndarray or None).
        iteration:       BO iteration at which this principle was discovered.
        source:          How this principle was generated:
                         "broad"    — LLM domain knowledge, no data
                         "refined"  — broad principle refined with training evidence
                         "oracle"   — extracted from real experimental data
                         "gp"       — extracted from GP (experiment agent) prediction
        gp_confidence:   GP model confidence (1 / (1 + std)) when source="gp".
                         Higher = more trustworthy. None for non-GP sources.
    """
    sequence: str
    outcome: Dict[str, float]
    primary_reward: float
    hypothesis: str
    principle_text: str
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    embedding: Optional[np.ndarray] = field(default=None, repr=False)
    iteration: int = 0
    source: str = "oracle"
    gp_confidence: Optional[float] = None

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        # numpy arrays are not JSON-serialisable; convert to list
        if self.embedding is not None:
            d["embedding"] = self.embedding.tolist()
        return d

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "Principle":
        emb = d.pop("embedding", None)
        # Handle older serialised principles without new fields
        d.setdefault("source", "oracle")
        d.setdefault("gp_confidence", None)
        p = cls(**d)
        if emb is not None:
            p.embedding = np.array(emb, dtype=np.float32)
        return p


class PrincipleBuffer:
    """
    In-memory store of discovered Principle objects.

    Mirrors PiFlow's ``PrincipleFlow.flow`` list but adds:
    - persistence (save / load JSON)
    - convenience accessors used by PrincipleScorer

    Usage::

        buffer = PrincipleBuffer()
        buffer.add(principle)
        rewards = buffer.rewards         # np.array of primary_reward values
        texts   = buffer.principle_texts # List[str]
        embeddings = buffer.embeddings   # np.ndarray (N, D)
    """

    def __init__(self) -> None:
        self._principles: List[Principle] = []

    # ------------------------------------------------------------------
    # Mutation
    # ------------------------------------------------------------------

    def add(self, principle: Principle) -> None:
        self._principles.append(principle)
        logger.info(
            "PrincipleBuffer | added principle #%d (reward=%.4f): %.80s…",
            len(self._principles),
            principle.primary_reward,
            principle.principle_text,
        )

    # ------------------------------------------------------------------
    # Read-only accessors
    # ------------------------------------------------------------------

    @property
    def size(self) -> int:
        return len(self._principles)

    def __len__(self) -> int:
        return self.size

    def get_all(self) -> List[Principle]:
        return list(self._principles)

    @property
    def rewards(self) -> np.ndarray:
        return np.array([p.primary_reward for p in self._principles], dtype=np.float64)

    @property
    def confidence_weighted_rewards(self) -> np.ndarray:
        """Rewards weighted by GP confidence. Non-GP principles get weight 1.0."""
        rewards = []
        for p in self._principles:
            confidence = p.gp_confidence if p.gp_confidence is not None else 1.0
            rewards.append(p.primary_reward * confidence)
        return np.array(rewards, dtype=np.float64)

    @property
    def principle_texts(self) -> List[str]:
        return [p.principle_text for p in self._principles]

    @property
    def hypotheses(self) -> List[str]:
        return [p.hypothesis for p in self._principles]

    @property
    def embeddings(self) -> Optional[np.ndarray]:
        """Return (N, D) array of embeddings; None if any is missing."""
        if not self._principles:
            return None
        embs = [p.embedding for p in self._principles]
        if any(e is None for e in embs):
            return None
        return np.stack(embs, axis=0).astype(np.float32)

    def best_principle(self) -> Optional[Principle]:
        if not self._principles:
            return None
        return max(self._principles, key=lambda p: p.primary_reward)

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, path: str) -> None:
        records = [p.to_dict() for p in self._principles]
        with open(path, "w") as f:
            json.dump(records, f, indent=2)
        logger.info("PrincipleBuffer saved %d principles to %s", self.size, path)

    def load(self, path: str) -> None:
        with open(path) as f:
            records = json.load(f)
        self._principles = [Principle.from_dict(r) for r in records]
        logger.info("PrincipleBuffer loaded %d principles from %s", self.size, path)

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------

    def summary(self) -> str:
        if not self._principles:
            return "PrincipleBuffer: empty"
        rewards = self.rewards
        lines = [
            f"PrincipleBuffer: {self.size} principles",
            f"  reward  min={rewards.min():.4f}  max={rewards.max():.4f}"
            f"  mean={rewards.mean():.4f}  std={rewards.std():.4f}",
        ]
        for i, p in enumerate(self._principles):
            lines.append(
                f"  [{i}] iter={p.iteration} reward={p.primary_reward:.4f}"
                f" | {p.principle_text[:80]}…"
            )
        return "\n".join(lines)
