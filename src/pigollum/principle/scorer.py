"""
PrincipleScorer: Exploration-exploitation scoring for principle-guided BO.

Directly adapts PiFlow's ``PrincipleFlow.suggest_action`` scoring logic:

    exploration_score[i]  = normalised semantic dissimilarity of principle i
                            (diverse principles score higher)
    exploitation_score[i] = sigmoid(z-score of reward[i])
                            (high-reward principles score higher)
    final_score[i]        = λ * exploration_score[i] + (1-λ) * exploitation_score[i]

The key extension over PiFlow is *candidate scoring*: instead of selecting
which known principle to focus on, we score unseen candidate sequences
based on their semantic similarity to the *best* known principles, so that
the BO optimizer can bias candidate selection toward regions consistent
with the most promising discovered principles.
"""
import logging
import math
from typing import List, Optional

import numpy as np

logger = logging.getLogger(__name__)


class PrincipleScorer:
    """
    Scores both known principles (for action selection) and unseen candidate
    sequences (for BO re-ranking).

    Parameters
    ----------
    embedding_model_name : str
        A sentence-transformers model name used to embed principle texts and
        candidate sequence descriptions.  Defaults to a small, fast model.
    lambda_factor : float
        Weight for exploration vs exploitation (0 = pure exploitation,
        1 = pure exploration).  Mirrors PiFlow's lambda_factor = 0.5.
    plateau_threshold : float
        If consecutive reward changes all fall below this, force exploration.
    plateau_count : int
        Number of consecutive small changes before plateau is declared.
    device : str
        Torch device for the embedding model ('cpu', 'cuda', …).
    """

    _DEFAULT_MODEL = "all-MiniLM-L6-v2"

    def __init__(
        self,
        embedding_model_name: str = _DEFAULT_MODEL,
        lambda_factor: float = 0.5,
        plateau_threshold: float = 0.1,
        plateau_count: int = 3,
        device: str = "cpu",
    ) -> None:
        self.lambda_factor = lambda_factor
        self.plateau_threshold = plateau_threshold
        self.plateau_count = plateau_count
        self.device = device
        self._model_name = embedding_model_name
        self._model = None  # lazy init to avoid slow startup
        self._recent_rewards: List[float] = []

    # ------------------------------------------------------------------
    # Lazy model loading
    # ------------------------------------------------------------------

    def _get_model(self):
        if self._model is None:
            try:
                from sentence_transformers import SentenceTransformer
                self._model = SentenceTransformer(self._model_name, device=self.device)
                logger.info("PrincipleScorer: loaded embedding model %s", self._model_name)
            except Exception as exc:
                logger.warning("Could not load sentence-transformers model: %s", exc)
        return self._model

    # ------------------------------------------------------------------
    # Embedding helpers
    # ------------------------------------------------------------------

    def embed(self, texts: List[str]) -> Optional[np.ndarray]:
        """Encode a list of texts → (N, D) float32 array, or None on failure."""
        model = self._get_model()
        if model is None or not texts:
            return None
        try:
            embs = model.encode(texts, convert_to_numpy=True, show_progress_bar=False)
            return embs.astype(np.float32)
        except Exception as exc:
            logger.warning("Embedding failed: %s", exc)
            return None

    def embed_single(self, text: str) -> Optional[np.ndarray]:
        result = self.embed([text])
        if result is None:
            return None
        return result[0]

    # ------------------------------------------------------------------
    # Cosine similarity utilities (pure numpy, mirrors PiFlow)
    # ------------------------------------------------------------------

    @staticmethod
    def _cosine_similarity_matrix(embeddings: np.ndarray) -> np.ndarray:
        """Compute (N, N) cosine similarity matrix."""
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True) + 1e-10
        normed = embeddings / norms
        return normed @ normed.T  # (N, N)

    @staticmethod
    def _cosine_similarity_cross(
        query: np.ndarray, keys: np.ndarray
    ) -> np.ndarray:
        """Compute cosine similarity between query (M, D) and keys (N, D) → (M, N)."""
        q_norm = query / (np.linalg.norm(query, axis=1, keepdims=True) + 1e-10)
        k_norm = keys  / (np.linalg.norm(keys,  axis=1, keepdims=True) + 1e-10)
        return q_norm @ k_norm.T  # (M, N)

    # ------------------------------------------------------------------
    # Principle scoring (mirrors PiFlow's suggest_action logic)
    # ------------------------------------------------------------------

    def score_principles(self, buffer) -> dict:
        """
        Score all principles in the buffer using PiFlow's algorithm.

        Returns a dict with keys: exploration, exploitation, final, best_idx,
        action_type, suggestion, plateau_detected.
        """
        from pigollum.principle.buffer import PrincipleBuffer  # avoid circular

        if buffer.size < 3:
            return {
                "best_idx": 0,
                "action_type": "explore",
                "plateau_detected": False,
                "suggestion": (
                    "[PiGollum] Initialize more observations to build principle knowledge. "
                    "Diverse exploration is crucial at this early stage."
                ),
                "exploration": {},
                "exploitation": {},
                "final": {},
            }

        rewards = buffer.rewards  # (N,)
        texts   = buffer.principle_texts  # List[str]

        # -------- Exploitation scores --------
        exploitation = self._compute_exploitation_scores(rewards)

        # -------- Exploration scores (semantic diversity) --------
        embs = buffer.embeddings
        if embs is None:
            embs = self.embed(texts)
        exploration = self._compute_exploration_scores(embs)

        # -------- Final scores --------
        final = self._compute_final_scores(exploration, exploitation)

        # -------- Decision --------
        best_idx = int(np.argmax(list(final.values())))
        best_principle = buffer.get_all()[best_idx]

        # Track recent rewards for plateau detection
        self._recent_rewards.append(best_principle.primary_reward)
        plateau = self._detect_plateau()

        if plateau:
            action_type = "explore"
            suggestion = (
                f"[PiGollum] Plateau detected over last {self.plateau_count} iterations "
                f"(reward change < {self.plateau_threshold}). "
                f"Forcing exploration: seek structurally diverse candidate sequences "
                f"that differ from previously evaluated ones. "
                f"Consider regions of sequence space not yet covered by known principles."
            )
        else:
            action_type, suggestion = self._determine_action(
                best_idx, best_principle, exploitation
            )

        return {
            "best_idx": best_idx,
            "action_type": action_type,
            "plateau_detected": plateau,
            "suggestion": suggestion,
            "exploration": exploration,
            "exploitation": exploitation,
            "final": final,
        }

    # ------------------------------------------------------------------
    # Candidate scoring for BO re-ranking (key extension over PiFlow)
    # ------------------------------------------------------------------

    def score_candidates(
        self,
        candidate_descriptions: List[str],
        buffer,
        top_k_principles: int = 5,
    ) -> np.ndarray:
        """
        Score M unseen candidate sequences by their similarity to the best
        known principles, using PiFlow's exploration-exploitation weighting.

        Parameters
        ----------
        candidate_descriptions : List[str]
            Short text descriptions of each candidate (e.g. from describe_sequence).
        buffer : PrincipleBuffer
            Current principle buffer.
        top_k_principles : int
            Only use the top-k principles (by reward) to avoid noise from bad ones.

        Returns
        -------
        scores : np.ndarray  shape (M,)
            Principle-guided score for each candidate ∈ [0, 1].
            Higher = more consistent with high-reward principles.
        """
        if buffer.size == 0 or not candidate_descriptions:
            return np.zeros(len(candidate_descriptions), dtype=np.float32)

        # --- Embed candidates ---
        cand_embs = self.embed(candidate_descriptions)  # (M, D)
        if cand_embs is None:
            logger.warning("PrincipleScorer: candidate embedding failed, returning zeros.")
            return np.zeros(len(candidate_descriptions), dtype=np.float32)

        # --- Select top-k principles by reward ---
        principles = buffer.get_all()
        rewards = buffer.rewards
        top_k = min(top_k_principles, len(principles))
        top_indices = np.argsort(rewards)[-top_k:]  # ascending, take last k
        top_principles = [principles[i] for i in top_indices]
        top_rewards = rewards[top_indices]

        # --- Embed principle texts ---
        top_texts = [p.principle_text for p in top_principles]
        if top_principles[0].embedding is not None:
            princ_embs = np.stack([p.embedding for p in top_principles], axis=0)
        else:
            princ_embs = self.embed(top_texts)

        if princ_embs is None:
            return np.zeros(len(candidate_descriptions), dtype=np.float32)

        # --- Cross-similarity: candidates vs top principles (M, k) ---
        sim_matrix = self._cosine_similarity_cross(cand_embs, princ_embs)  # (M, k)

        # --- Weight principles by normalised reward ---
        reward_weights = top_rewards - top_rewards.min() + 1e-8
        reward_weights = reward_weights / reward_weights.sum()

        # --- Weighted similarity: (M,) ---
        exploitation_sim = sim_matrix @ reward_weights  # similarity to best principles

        # --- Diversity bonus: penalise candidates similar to already-seen sequences ---
        all_texts = buffer.principle_texts
        all_embs = buffer.embeddings
        if all_embs is None:
            all_embs = self.embed(all_texts)

        if all_embs is not None:
            all_sim = self._cosine_similarity_cross(cand_embs, all_embs)  # (M, N)
            avg_sim_to_known = all_sim.mean(axis=1)  # (M,)
            exploration_div = 1.0 - avg_sim_to_known
        else:
            exploration_div = np.ones(len(candidate_descriptions), dtype=np.float32)

        # --- Combine: λ * diversity + (1-λ) * similarity-to-best ---
        lam = self.lambda_factor
        combined = lam * exploration_div + (1.0 - lam) * exploitation_sim

        # Normalise to [0, 1]
        lo, hi = combined.min(), combined.max()
        if hi - lo > 1e-8:
            combined = (combined - lo) / (hi - lo)

        return combined.astype(np.float32)

    # ------------------------------------------------------------------
    # Private scoring methods (mirror PiFlow internals)
    # ------------------------------------------------------------------

    def _compute_exploitation_scores(self, rewards: np.ndarray) -> dict:
        n = len(rewards)
        scores = {}
        if n == 0:
            return scores
        mean, std = rewards.mean(), rewards.std()
        for i, r in enumerate(rewards):
            if std > 1e-8:
                z = (r - mean) / std
            else:
                z = 0.0
            scores[i] = 1.0 / (1.0 + math.exp(-z))  # sigmoid
        return scores

    def _compute_exploration_scores(self, embeddings: Optional[np.ndarray]) -> dict:
        if embeddings is None or len(embeddings) == 0:
            return {}
        n = len(embeddings)
        if n <= 3:
            return {i: 1.0 for i in range(n)}

        sim = self._cosine_similarity_matrix(embeddings)  # (N, N)
        scores = {}
        for i in range(n):
            row = np.concatenate([sim[i, :i], sim[i, i+1:]])  # exclude self
            avg_sim = row.mean()
            scores[i] = float(1.0 - avg_sim)  # dissimilarity

        # Normalise to [0, 1]
        vals = np.array(list(scores.values()))
        lo, hi = vals.min(), vals.max()
        if hi - lo > 1e-8:
            for i in scores:
                scores[i] = (scores[i] - lo) / (hi - lo)
        return scores

    def _compute_final_scores(self, exploration: dict, exploitation: dict) -> dict:
        final = {}
        lam = self.lambda_factor
        for i in exploration:
            final[i] = lam * exploration.get(i, 0.5) + (1 - lam) * exploitation.get(i, 0.5)
        return final

    def _detect_plateau(self) -> bool:
        if len(self._recent_rewards) < self.plateau_count:
            return False
        recent = self._recent_rewards[-self.plateau_count:]
        return all(
            abs(recent[j] - recent[j - 1]) < self.plateau_threshold
            for j in range(1, len(recent))
        )

    @staticmethod
    def _determine_action(best_idx: int, best_principle, exploitation: dict):
        score = exploitation.get(best_idx, 0.5)
        if score > 0.7:
            return "refine", (
                f"[PiGollum] Refine around principle: '{best_principle.principle_text[:100]}…'. "
                f"This principle achieved reward={best_principle.primary_reward:.4f} and shows "
                f"high exploitation potential. Seek sequences similar to this variant."
            )
        elif score > 0.4:
            return "validate", (
                f"[PiGollum] Validate principle: '{best_principle.principle_text[:100]}…'. "
                f"Reward={best_principle.primary_reward:.4f} shows moderate promise. "
                f"Confirm with structurally similar but distinct sequences."
            )
        else:
            return "explore", (
                f"[PiGollum] Explore: current best principle "
                f"(reward={best_principle.primary_reward:.4f}) has low exploitation score. "
                f"Diversify into unexplored regions of sequence space."
            )
