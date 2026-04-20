"""
PrincipleJournal: tracks the full lineage of discovered principles.

Records a timestamped snapshot at every key event:
  - WARM_START  : principles extracted from the initial training set
  - ITERATION   : one BO iteration (action decision + new principle found)

At the end of an experiment, call ``journal.report()`` to print a formatted
timeline, and ``journal.save(path)`` to write it to JSON.

The final section of the report highlights the "winning" principles:
  - Top-k by primary reward (best outcomes observed)
  - Top-k by selection count (most often chosen as guiding principle)
  - Top-k by combined influence score (reward × selection_count)
"""
import json
import logging
import os
import textwrap
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional, Any

import numpy as np

logger = logging.getLogger(__name__)

# Terminal width for pretty-printing
_W = 78


# ─────────────────────────────────────────────────────────────────────────────
# Data structures
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class PrincipleSnapshot:
    """
    State of a single principle at a given iteration.

    Captured when PrincipleScorer.score_principles() runs so we can replay
    how scores evolved.
    """
    principle_id:    str
    iteration_added: int          # BO iteration when first discovered
    source:          str          # "broad" | "refined" | "gp" | "oracle"
    principle_text:  str
    hypothesis:      str
    reward:          float
    exploration_score: float      # 0–1, semantic dissimilarity
    exploitation_score: float     # 0–1, sigmoid z-score
    final_score:     float        # λ*explore + (1-λ)*exploit
    was_selected:    bool         # True if this was the "best" this iteration


@dataclass
class IterationRecord:
    """Full record of one BO iteration."""
    iteration:         int          # -1 = warm-start
    phase:             str          # "warm_start" | "bo"
    n_principles_before: int
    action_type:       str          # "explore" | "refine" | "validate" | "pure_bo"
    plateau_detected:  bool
    suggestion:        str
    principle_scores:  List[PrincipleSnapshot]   # scores of ALL principles this iter
    best_principle_id: Optional[str]             # ID of the selected principle
    # Newly discovered principle this iteration (None for warm-start records
    # where we store principles individually below)
    new_principle_id:  Optional[str]
    new_principle_text: Optional[str]
    new_principle_reward: Optional[float]
    # BO outcome
    selected_sequences:  List[str]
    selected_outcomes:   List[Dict[str, float]]  # one dict per selected sequence
    best_train_y:        Optional[List[float]]   # best-so-far per objective


# ─────────────────────────────────────────────────────────────────────────────
# Journal
# ─────────────────────────────────────────────────────────────────────────────

class PrincipleJournal:
    """
    Records and reports on the full evolution of principles over a PiGollum run.

    Usage
    -----
    Attach to PiGollumOptimizer via ``optimizer.journal = PrincipleJournal()``,
    then call the record_* methods from run_experiment.py, and finally call
    ``report()`` at the end.
    """

    def __init__(self, objective_names: List[str]) -> None:
        self.objective_names = objective_names
        self._records: List[IterationRecord] = []
        # selection_counts[principle_id] = number of times chosen as best
        self._selection_counts: Dict[str, int] = {}

    # ------------------------------------------------------------------
    # Recording API
    # ------------------------------------------------------------------

    def record_iteration(
        self,
        iteration: int,
        phase: str,
        n_principles_before: int,
        action_info: Dict[str, Any],           # from PrincipleScorer.score_principles
        selected_sequences: List[str],
        selected_outcomes: List[Dict[str, float]],
        best_train_y: Optional[List[float]],
        new_principle_id: Optional[str] = None,
        new_principle_text: Optional[str] = None,
        new_principle_reward: Optional[float] = None,
        buffer=None,                           # PrincipleBuffer (for snapshots)
    ) -> None:
        """Record one BO iteration."""
        best_id = None
        snapshots: List[PrincipleSnapshot] = []

        if buffer is not None and buffer.size > 0:
            principles = buffer.get_all()
            expl  = action_info.get("exploration",  {})
            explt = action_info.get("exploitation", {})
            final = action_info.get("final",        {})
            best_idx = action_info.get("best_idx",  0)
            best_id  = principles[best_idx].id if 0 <= best_idx < len(principles) else None

            if best_id:
                self._selection_counts[best_id] = self._selection_counts.get(best_id, 0) + 1

            for i, p in enumerate(principles):
                snapshots.append(PrincipleSnapshot(
                    principle_id=p.id,
                    iteration_added=p.iteration,
                    source=p.source,
                    principle_text=p.principle_text,
                    hypothesis=p.hypothesis,
                    reward=p.primary_reward,
                    exploration_score=float(expl.get(i,  0.0)),
                    exploitation_score=float(explt.get(i, 0.0)),
                    final_score=float(final.get(i,        0.0)),
                    was_selected=(i == best_idx),
                ))

        rec = IterationRecord(
            iteration=iteration,
            phase=phase,
            n_principles_before=n_principles_before,
            action_type=action_info.get("action_type", "pure_bo"),
            plateau_detected=action_info.get("plateau_detected", False),
            suggestion=action_info.get("suggestion", ""),
            principle_scores=snapshots,
            best_principle_id=best_id,
            new_principle_id=new_principle_id,
            new_principle_text=new_principle_text,
            new_principle_reward=new_principle_reward,
            selected_sequences=list(selected_sequences),
            selected_outcomes=list(selected_outcomes),
            best_train_y=best_train_y,
        )
        self._records.append(rec)

    # ------------------------------------------------------------------
    # Analysis helpers
    # ------------------------------------------------------------------

    def winning_principles(self, buffer, top_k: int = 5) -> Dict[str, List]:
        """
        Return three ranked lists of winning principles.

        Returns
        -------
        dict with keys:
          "by_reward"    : top_k principles sorted by primary_reward (desc)
          "by_selection" : top_k principles sorted by selection count (desc)
          "by_influence" : top_k principles sorted by reward × selection_count (desc)
        """
        if buffer is None or buffer.size == 0:
            return {"by_reward": [], "by_selection": [], "by_influence": []}

        principles = buffer.get_all()
        rows = []
        for p in principles:
            sel = self._selection_counts.get(p.id, 0)
            rows.append({
                "principle": p,
                "selection_count": sel,
                "influence": p.primary_reward * (sel + 1),  # +1 avoids zero
            })

        by_reward    = sorted(rows, key=lambda r: r["principle"].primary_reward, reverse=True)[:top_k]
        by_selection = sorted(rows, key=lambda r: r["selection_count"],          reverse=True)[:top_k]
        by_influence = sorted(rows, key=lambda r: r["influence"],                reverse=True)[:top_k]

        return {
            "by_reward":    by_reward,
            "by_selection": by_selection,
            "by_influence": by_influence,
        }

    # ------------------------------------------------------------------
    # Reporting
    # ------------------------------------------------------------------

    def report(self, buffer=None, top_k: int = 5) -> str:
        """
        Generate and return the full human-readable evolution report.
        Also prints it to stdout.
        """
        lines = []
        _h1 = lambda t: lines.append("\n" + "═" * _W + f"\n  {t}\n" + "═" * _W)
        _h2 = lambda t: lines.append("\n" + "─" * _W + f"\n  {t}\n" + "─" * _W)
        _p  = lambda t, indent=2: lines.extend(
            textwrap.wrap(t, width=_W - indent, initial_indent=" " * indent,
                          subsequent_indent=" " * (indent + 2))
        )

        _h1("PiGollum Principle Evolution Report")

        # ── Warm-start section ──────────────────────────────────────────
        warmup = [r for r in self._records if r.phase == "warm_start"]
        if warmup:
            _h2(f"Phase 0 — Warm-start  ({len(warmup)} initial principles)")
            for r in warmup:
                for snap in r.principle_scores:
                    if snap.iteration_added == -1:
                        lines.append(
                            f"  [warm]  reward={snap.reward:+.4f}"
                            f"  expl={snap.exploration_score:.3f}"
                            f"  expt={snap.exploitation_score:.3f}"
                        )
                        _p(f"Principle: {snap.principle_text}", indent=10)

        # ── BO iterations ───────────────────────────────────────────────
        bo_records = [r for r in self._records if r.phase == "bo"]
        for r in bo_records:
            plateau_tag = "  ⚠ PLATEAU" if r.plateau_detected else ""
            _h2(
                f"Iteration {r.iteration}  "
                f"[{r.action_type.upper()}{plateau_tag}]  "
                f"— {r.n_principles_before} principles in buffer"
            )

            # Action suggestion
            lines.append(f"  Guidance: {r.suggestion[:120]}…" if len(r.suggestion) > 120
                         else f"  Guidance: {r.suggestion}")

            # Score table for all principles
            if r.principle_scores:
                lines.append("")
                lines.append(
                    f"  {'#':>3}  {'iter':>4}  {'reward':>8}  "
                    f"{'expl':>6}  {'expt':>6}  {'final':>6}  {'selected':>8}  principle"
                )
                lines.append("  " + "-" * (_W - 4))
                for i, s in enumerate(r.principle_scores):
                    sel_mark = "  ★" if s.was_selected else ""
                    short = s.principle_text[:50].replace("\n", " ")
                    lines.append(
                        f"  {i:>3}  {s.iteration_added:>4}  {s.reward:>+8.4f}  "
                        f"{s.exploration_score:>6.3f}  {s.exploitation_score:>6.3f}  "
                        f"{s.final_score:>6.3f}  {sel_mark:>8}  {short}…"
                    )

            # Newly discovered principle
            if r.new_principle_text:
                lines.append("")
                lines.append(f"  ► New principle discovered  (reward={r.new_principle_reward:+.4f})")
                _p(r.new_principle_text, indent=6)

            # BO outcome
            if r.selected_outcomes:
                lines.append("")
                for seq, out in zip(r.selected_sequences, r.selected_outcomes):
                    out_str = "  ".join(f"{k}={v:+.4f}" for k, v in out.items())
                    lines.append(f"  Evaluated: {seq[:50]}…  →  {out_str}")
            if r.best_train_y:
                best_str = "  ".join(
                    f"{n}={v:+.4f}"
                    for n, v in zip(self.objective_names, r.best_train_y)
                )
                lines.append(f"  Best so far: {best_str}")

        # ── Winning principles ──────────────────────────────────────────
        _h1("Winning Principles")

        if buffer is not None:
            winners = self.winning_principles(buffer, top_k=top_k)

            def _print_ranked(title, rows, key_label, key_fn):
                _h2(title)
                for rank, row in enumerate(rows, 1):
                    p = row["principle"]
                    lines.append(
                        f"  #{rank}  reward={p.primary_reward:+.4f}  "
                        f"{key_label}={key_fn(row)}  "
                        f"iter={p.iteration}  id={p.id[:8]}…"
                    )
                    _p(f"Principle: {p.principle_text}", indent=6)
                    _p(f"Hypothesis: {p.hypothesis[:200]}", indent=6)
                    lines.append("")

            _print_ranked(
                f"Top {top_k} by Observed Reward (best outcomes)",
                winners["by_reward"], "reward",
                lambda r: f"{r['principle'].primary_reward:+.4f}",
            )
            _print_ranked(
                f"Top {top_k} by Selection Count (most often guided BO)",
                winners["by_selection"], "selections",
                lambda r: str(r["selection_count"]),
            )
            _print_ranked(
                f"Top {top_k} by Influence Score (reward × selections)",
                winners["by_influence"], "influence",
                lambda r: f"{r['influence']:.2f}",
            )

            # Highlight the single overall winner
            overall = winners["by_influence"][0] if winners["by_influence"] else None
            if overall:
                _h1("The Winning Principle")
                p = overall["principle"]
                lines.append(f"  Reward:         {p.primary_reward:+.4f}")
                lines.append(f"  Selection count: {self._selection_counts.get(p.id, 0)}")
                lines.append(f"  Influence score: {overall['influence']:.2f}")
                lines.append(f"  Discovered at iteration: {p.iteration}")
                lines.append(f"  Sequence: {p.sequence[:80]}…")
                lines.append("")
                lines.append("  HYPOTHESIS")
                _p(p.hypothesis, indent=4)
                lines.append("")
                lines.append("  PRINCIPLE")
                _p(p.principle_text, indent=4)

        report_str = "\n".join(lines)
        print(report_str)
        return report_str

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, path: str, buffer=None) -> None:
        """Save the full journal + winning principles to a JSON file."""
        def _snap_dict(s: PrincipleSnapshot) -> dict:
            return asdict(s)

        def _rec_dict(r: IterationRecord) -> dict:
            d = asdict(r)
            d["principle_scores"] = [_snap_dict(s) for s in r.principle_scores]
            return d

        payload = {
            "objective_names": self.objective_names,
            "selection_counts": self._selection_counts,
            "records": [_rec_dict(r) for r in self._records],
        }
        if buffer is not None:
            winners = self.winning_principles(buffer)
            payload["winning_principles"] = {
                k: [
                    {
                        "principle_id":   row["principle"].id,
                        "principle_text": row["principle"].principle_text,
                        "hypothesis":     row["principle"].hypothesis,
                        "sequence":       row["principle"].sequence,
                        "reward":         row["principle"].primary_reward,
                        "iteration":      row["principle"].iteration,
                        "selection_count": row["selection_count"],
                        "influence":      row["influence"],
                    }
                    for row in rows
                ]
                for k, rows in winners.items()
            }

        with open(path, "w") as f:
            json.dump(payload, f, indent=2)
        logger.info("PrincipleJournal saved to %s", path)

    def save_text_report(self, path: str, buffer=None) -> None:
        """Write the human-readable report to a .txt file."""
        text = self.report(buffer=buffer)
        with open(path, "w") as f:
            f.write(text)
        logger.info("Text report saved to %s", path)
