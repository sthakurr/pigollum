"""
run_experiment.py – PiGollum experiment entry point for the biocat dataset.

Redesigned architecture: GP acts as the experiment agent (oracle) that
validates LLM-generated hypotheses in an inner PiFlow loop.

Usage
-----
    conda run -n gollum python run_experiment.py --config configs/biocat_pigollum.yaml

Optional LLM integration (principle extraction via GPT/local LLM):
    export PIGOLLUM_LLM_API_KEY=sk-...
    export PIGOLLUM_LLM_BASE_URL=http://localhost:11434/v1
    export PIGOLLUM_LLM_MODEL=gpt-4o-mini

Architecture
────────────
    Warm-start:
        1. LLM generates broad domain-knowledge principles
        2. Retrieve relevant training examples for each
        3. LLM refines principles with supporting/contradicting evidence

    Per BO iteration:
        1. Train GP surrogate
        2. GP predicts all candidates (means + stds)
        3. Inner PiFlow loop (n_inner_steps):
           a. Planner suggests action (explore/refine/validate)
           b. LLM generates predictive hypotheses for selected candidates
           c. GP validates hypotheses (provides predictions)
           d. Extract principles from (hypothesis, GP prediction)
        4. Combine acquisition scores + principle scores
        5. Select candidate for real oracle evaluation
        6. Extract principle from real outcome

Outputs
-------
    bo_results.json        – per-iteration BO outcomes
    principles.json        – final principle buffer
    journal.json           – full principle evolution lineage
    evolution_report.txt   – human-readable evolution report
"""
import argparse
import json
import logging
import os
import sys
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import yaml
from tqdm import tqdm

# ── gollum path setup ────────────────────────────────────────────────────────
_HERE = Path(__file__).parent.resolve()
_GOLLUM_SRC = _HERE / "gollum-2" / "src"
if _GOLLUM_SRC.is_dir():
    sys.path.insert(0, str(_GOLLUM_SRC))

# ── pigollum path setup ───────────────────────────────────────────────────────
_PIGOLLUM_SRC = _HERE / "src"
if _PIGOLLUM_SRC.is_dir():
    sys.path.insert(0, str(_PIGOLLUM_SRC))

from gollum.data.module import BaseDataModule
from gollum.utils.config import instantiate_class
from gollum.data.utils import torch_delete_rows
from botorch.utils.multi_objective.pareto import is_non_dominated

from pigollum.bo.pi_optimizer import PiGollumOptimizer
from pigollum.principle.buffer import PrincipleBuffer
from pigollum.principle.extractor import PrincipleExtractor
from pigollum.principle.scorer import PrincipleScorer
from pigollum.principle.planner import Planner
from pigollum.principle.journal import PrincipleJournal

warnings.filterwarnings("ignore")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("pigollum.run_experiment")


# ─────────────────────────────────────────────────────────────────────────────
# Config loading
# ─────────────────────────────────────────────────────────────────────────────

def load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def build_data_module(cfg: dict, data_root: str) -> BaseDataModule:
    """Instantiate BaseDataModule, resolving relative data paths."""
    data_cfg = cfg["data"].copy()
    init_args = data_cfg["init_args"]

    for key in ("data_path", "test_data_path", "round_2_data_path"):
        if key in init_args and init_args[key] is not None:
            p = Path(init_args[key])
            if not p.is_absolute():
                init_args[key] = str(Path(data_root) / p)

    return instantiate_class(data_cfg)


def build_pi_optimizer(cfg: dict, extractor: Optional[PrincipleExtractor]) -> PiGollumOptimizer:
    """Build PiGollumOptimizer from config."""
    bo_cfg = cfg["bo"]
    acq_cfg = cfg["acquisition"]
    surr_cfg = cfg["surrogate_model"]

    pi_cfg = cfg.get("pigollum", {})
    principle_weight = pi_cfg.get("principle_weight", 0.3)
    min_principles = pi_cfg.get("min_principles_for_guidance", 3)
    weight_schedule = pi_cfg.get("principle_weight_schedule", None)
    embedding_model = pi_cfg.get("embedding_model", "all-MiniLM-L6-v2")
    n_inner_steps = pi_cfg.get("n_inner_steps", 5)
    candidate_sample_size = pi_cfg.get("candidate_sample_size", 10)

    device_str = surr_cfg.get("init_args", {}).get("device", "cpu")

    scorer = PrincipleScorer(
        embedding_model_name=embedding_model,
        lambda_factor=pi_cfg.get("lambda_factor", 0.5),
        plateau_threshold=pi_cfg.get("plateau_threshold", 0.1),
        plateau_count=pi_cfg.get("plateau_count", 3),
        device=device_str,
    )

    planner = Planner(
        n_inner_steps=n_inner_steps,
        candidate_sample_size=candidate_sample_size,
        task_context=pi_cfg.get("task_context", ""),
    )

    batch_size = bo_cfg.get("init_args", {}).get("batch_size", 1)

    optimizer = PiGollumOptimizer(
        surrogate_model_config=surr_cfg,
        acq_function_config=acq_cfg,
        batch_size=batch_size,
        principle_weight=principle_weight,
        min_principles_for_guidance=min_principles,
        principle_weight_schedule=weight_schedule,
        n_inner_steps=n_inner_steps,
        candidate_sample_size=candidate_sample_size,
        extractor=extractor,
        scorer=scorer,
        planner=planner,
    )
    return optimizer


# ─────────────────────────────────────────────────────────────────────────────
# BO loop helpers
# ─────────────────────────────────────────────────────────────────────────────

def get_candidate_sequences(
    candidate_indices: List[int],
    all_sequences: List[str],
    heldout_global_indices: torch.Tensor,
) -> List[str]:
    """Map local design-space indices → raw sequence strings."""
    seqs = []
    for local_idx in range(len(heldout_global_indices)):
        global_idx = int(heldout_global_indices[local_idx].item())
        seqs.append(all_sequences[global_idx])
    return seqs


def log_iteration(
    iteration: int,
    newly_selected_seqs: List[str],
    newly_selected_y: torch.Tensor,
    train_y: torch.Tensor,
    principle_buffer: PrincipleBuffer,
    objective_names: List[str],
    results: List[dict],
) -> None:
    """Print and record iteration statistics."""
    new_y_np = newly_selected_y.cpu().numpy()
    train_y_np = train_y.cpu().numpy()

    best_so_far = train_y_np.max(axis=0)

    dominated_mask = is_non_dominated(torch.tensor(train_y_np))
    pareto_size = int(dominated_mask.sum().item())

    # Count principles by source
    all_principles = principle_buffer.get_all()
    n_broad = sum(1 for p in all_principles if p.source == "broad")
    n_refined = sum(1 for p in all_principles if p.source == "refined")
    n_gp = sum(1 for p in all_principles if p.source == "gp")
    n_oracle = sum(1 for p in all_principles if p.source == "oracle")

    print(f"\n{'─'*60}")
    print(f"  Iteration {iteration} Summary")
    print(f"{'─'*60}")
    for i, obj in enumerate(objective_names):
        new_vals = new_y_np[:, i] if new_y_np.ndim > 1 else new_y_np
        print(f"  {obj}: new={new_vals[0]:.4f}  best_so_far={best_so_far[i]:.4f}")
    print(f"  Pareto front size: {pareto_size}")
    print(f"  Principles: {principle_buffer.size} total "
          f"(broad={n_broad}, refined={n_refined}, gp={n_gp}, oracle={n_oracle})")
    if principle_buffer.size > 0:
        best_p = principle_buffer.best_principle()
        print(f"  Best principle (reward={best_p.primary_reward:.4f}, source={best_p.source}):")
        print(f"    {best_p.principle_text[:120]}…")
    print(f"{'─'*60}\n")

    results.append({
        "iteration": iteration,
        "newly_selected_sequences": newly_selected_seqs,
        "newly_selected_y": new_y_np.tolist(),
        "best_so_far": best_so_far.tolist(),
        "pareto_size": pareto_size,
        "n_principles": principle_buffer.size,
        "n_principles_by_source": {
            "broad": n_broad, "refined": n_refined,
            "gp": n_gp, "oracle": n_oracle,
        },
    })


# ─────────────────────────────────────────────────────────────────────────────
# Main experiment loop
# ─────────────────────────────────────────────────────────────────────────────

def run(cfg: dict, data_root: str, output_dir: str, seed: int) -> None:
    torch.manual_seed(seed)
    np.random.seed(seed)

    os.makedirs(output_dir, exist_ok=True)

    # ------------------------------------------------------------------
    # 1. Data module
    # ------------------------------------------------------------------
    logger.info("Loading biocat dataset…")
    dm = build_data_module(cfg, data_root)

    input_col = cfg["data"]["init_args"]["input_column"]
    all_sequences: List[str] = dm.data[input_col].tolist()

    objective_names: List[str] = cfg["data"]["init_args"]["target_column"]
    if isinstance(objective_names, str):
        objective_names = [objective_names]

    logger.info(
        "Dataset: %d total, %d train, %d heldout",
        len(dm.data), len(dm.train_indexes), len(dm.heldout_indices),
    )

    # ------------------------------------------------------------------
    # 2. PrincipleExtractor (wraps LLM)
    # ------------------------------------------------------------------
    pi_cfg = cfg.get("pigollum", {})
    task_context = pi_cfg.get(
        "task_context",
        (
            "Biocatalytic enzyme engineering: optimise amino acid sequences (enzymes) to maximise "
            "reaction yield and enantioselectivity (ddg_scaled) for a target asymmetric "
            "transformation: CC(C(=O)ON1C(=O)C2=C(C=CC=C2)C1=O)C1=CC=CC=C1 + TMSCN --> CC(C#N)C1=CC=CC=C1. "
            "The sequences are not really related to each other as we are in an exploration phase but we need "
            "to train an intelligent GP that can learn the defining principles for this task from the "
            "weak signal we have. We need the GP to have enough information to rank a subset of sequences "
            "to perform virtual screening. The top-ranked sequences will go further for experimental validation."
        ),
    )

    _dtype_map = {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32,
    }
    hf_dtype = _dtype_map.get(pi_cfg.get("hf_torch_dtype", "bfloat16"), torch.bfloat16)

    extractor = PrincipleExtractor(
        task_context=task_context,
        objective_names=objective_names,
        hf_model_name=pi_cfg.get("hf_model_name", "Qwen/Qwen2.5-7B-Instruct"),
        torch_dtype=hf_dtype,
        backend=pi_cfg.get("backend", None),
        llm_api_key=pi_cfg.get("llm_api_key"),
        llm_base_url=pi_cfg.get("llm_base_url"),
        llm_model=pi_cfg.get("llm_model"),
    )

    # ------------------------------------------------------------------
    # 3. PiGollum optimizer + journal
    # ------------------------------------------------------------------
    optimizer = build_pi_optimizer(cfg, extractor)
    journal = PrincipleJournal(objective_names=objective_names)
    optimizer.journal = journal

    # ------------------------------------------------------------------
    # 4. BO state initialisation
    # ------------------------------------------------------------------
    train_x = dm.train_x.clone()
    train_y = dm.train_y.clone()
    heldout_x = dm.heldout_x.clone()
    heldout_y = dm.heldout_y.clone()
    heldout_global_indices = dm.heldout_indices.clone()

    n_iters = cfg.get("n_iters", 5)
    results: List[dict] = []

    # ------------------------------------------------------------------
    # 5. Warm-start: broad principles → evidence-based refinement
    # ------------------------------------------------------------------
    logger.info("Starting PiGollum BO loop (%d iterations)…", n_iters)

    if pi_cfg.get("warm_start_principles", True):
        logger.info("=== Warm-start: Broad → Refined principle generation ===")

        # Build training outcomes
        train_seqs = [all_sequences[int(idx)] for idx in dm.train_indexes]
        train_y_np = train_y.cpu().numpy()
        train_outcomes = [
            {name: float(train_y_np[i, j]) for j, name in enumerate(objective_names)}
            for i in range(len(train_seqs))
        ]

        n_broad = pi_cfg.get("n_broad_principles", 5)

        optimizer.warm_start_with_refinement(
            train_sequences=train_seqs,
            train_outcomes=train_outcomes,
            n_broad_principles=n_broad,
        )

        # Record warm-start snapshot in journal
        ws_action = optimizer.principle_scorer.score_principles(optimizer.principle_buffer)
        optimizer._last_action_info = ws_action
        journal.record_iteration(
            iteration=-1,
            phase="warm_start",
            n_principles_before=0,
            action_info=ws_action,
            selected_sequences=[],
            selected_outcomes=[],
            best_train_y=train_y.cpu().numpy().max(axis=0).tolist(),
            buffer=optimizer.principle_buffer,
        )
        logger.info(
            "Warm-start complete: %d principles in buffer", optimizer.principle_buffer.size
        )

    # ------------------------------------------------------------------
    # 6. Main BO loop with inner PiFlow loop
    # ------------------------------------------------------------------
    for iteration in range(1, n_iters + 1):
        logger.info("\n=== Iteration %d / %d ===", iteration, n_iters)

        if heldout_x.size(0) == 0:
            logger.warning("Design space exhausted – stopping early.")
            break

        n_principles_before = optimizer.principle_buffer.size

        # --- Build candidate sequence list ---
        candidate_seqs: List[str] = [
            all_sequences[int(heldout_global_indices[i].item())]
            for i in range(heldout_global_indices.size(0))
        ]

        # --- Suggest next experiments ---
        # This now runs: train GP → GP predictions → inner PiFlow loop →
        # acquisition + principle scores → selection
        candidates = optimizer.suggest_next_experiments(
            train_x=train_x,
            train_y=train_y,
            design_space=heldout_x,
            candidate_sequences=candidate_seqs,
            objective_names=objective_names,
        )

        selected_local_indices = optimizer._last_selected_indices
        action_info = optimizer._last_action_info

        # --- Oracle evaluation (look up true labels) ---
        newly_selected_seqs: List[str] = []
        newly_selected_y_rows: List[torch.Tensor] = []

        for local_idx in selected_local_indices:
            global_idx = int(heldout_global_indices[local_idx].item())
            seq = all_sequences[global_idx]
            y_row = heldout_y[local_idx]
            newly_selected_seqs.append(seq)
            newly_selected_y_rows.append(y_row)

        newly_selected_y = torch.stack(newly_selected_y_rows, dim=0)

        # --- Update training data ---
        new_x = torch.stack([heldout_x[i] for i in selected_local_indices], dim=0)
        train_x = torch.cat([train_x, new_x], dim=0)
        train_y = torch.cat([train_y, newly_selected_y], dim=0)

        # --- Remove evaluated points from design space ---
        keep_mask = torch.ones(heldout_x.size(0), dtype=torch.bool)
        for local_idx in selected_local_indices:
            keep_mask[local_idx] = False
        heldout_x = heldout_x[keep_mask]
        heldout_y = heldout_y[keep_mask]
        heldout_global_indices = heldout_global_indices[keep_mask]

        # --- Extract principles from REAL oracle evaluation ---
        new_y_np = newly_selected_y.cpu().numpy()
        newly_extracted_principles = []
        for seq, y_row in zip(newly_selected_seqs, new_y_np):
            outcome = {name: float(y_row[j]) for j, name in enumerate(objective_names)}
            p = optimizer.update_principles(seq, outcome, iteration=iteration, source="oracle")
            if p is not None:
                newly_extracted_principles.append(p)

        # --- Record in journal ---
        selected_outcomes = [
            {name: float(new_y_np[k, j]) for j, name in enumerate(objective_names)}
            for k in range(len(newly_selected_seqs))
        ]
        first_new = newly_extracted_principles[0] if newly_extracted_principles else None
        journal.record_iteration(
            iteration=iteration,
            phase="bo",
            n_principles_before=n_principles_before,
            action_info=action_info,
            selected_sequences=newly_selected_seqs,
            selected_outcomes=selected_outcomes,
            best_train_y=train_y.cpu().numpy().max(axis=0).tolist(),
            new_principle_id=first_new.id if first_new else None,
            new_principle_text=first_new.principle_text if first_new else None,
            new_principle_reward=first_new.primary_reward if first_new else None,
            buffer=optimizer.principle_buffer,
        )

        # --- Console summary ---
        log_iteration(
            iteration=iteration,
            newly_selected_seqs=newly_selected_seqs,
            newly_selected_y=newly_selected_y,
            train_y=train_y,
            principle_buffer=optimizer.principle_buffer,
            objective_names=objective_names,
            results=results,
        )

    # ------------------------------------------------------------------
    # 7. Save outputs
    # ------------------------------------------------------------------
    results_path = os.path.join(output_dir, "bo_results.json")
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    logger.info("Results saved to %s", results_path)

    principles_path = os.path.join(output_dir, "principles.json")
    optimizer.principle_buffer.save(principles_path)
    logger.info("Principles saved to %s", principles_path)

    # ── Evolution report ────────────────────────────────────────────────
    print("\n" + optimizer.principle_buffer.summary())

    report_txt_path = os.path.join(output_dir, "evolution_report.txt")
    journal.save_text_report(report_txt_path, buffer=optimizer.principle_buffer)
    logger.info("Evolution report saved to %s", report_txt_path)

    journal_path = os.path.join(output_dir, "journal.json")
    journal.save(journal_path, buffer=optimizer.principle_buffer)
    logger.info("Journal saved to %s", journal_path)

    # Final best-observed
    final_y = train_y.cpu().numpy()
    best = final_y.max(axis=0)
    print("\nFinal best observed:")
    for name, val in zip(objective_names, best):
        print(f"  {name}: {val:.4f}")


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="PiGollum: Principle-Guided BO with GP as Experiment Agent"
    )
    parser.add_argument(
        "--config", default="configs/biocat_pigollum.yaml",
        help="Path to YAML config file",
    )
    parser.add_argument(
        "--data_root", default="gollum-2",
        help="Root directory containing the data/ folder (e.g. gollum-2/)",
    )
    parser.add_argument(
        "--output_dir", default="results/biocat_pigollum",
        help="Directory to save results and principles",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--n_iters", type=int, default=None,
        help="Override n_iters from config",
    )
    args = parser.parse_args()

    cfg = load_config(args.config)
    if args.n_iters is not None:
        cfg["n_iters"] = args.n_iters

    run(
        cfg=cfg,
        data_root=args.data_root,
        output_dir=args.output_dir,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
