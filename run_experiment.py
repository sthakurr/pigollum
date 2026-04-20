"""
run_experiment.py – PiGollum experiment entry point.

Usage
-----
    conda run -n gollum python run_experiment.py --config configs/biocat_pigollum.yaml

Optional LLM integration:
    export PIGOLLUM_LLM_API_KEY=sk-...
    export PIGOLLUM_LLM_BASE_URL=http://localhost:11434/v1
    export PIGOLLUM_LLM_MODEL=gpt-4o-mini

Architecture
────────────
    Warm-start (once):
        1. LLM generates broad domain-knowledge principles
        2. LLM refines each principle with supporting/contradicting evidence

    Per BO iteration:
        1. Fine-tune LLM-GP on (train_x, train_y)
        2. GP predicts all candidates (means + stds)
        3. Compute acquisition scores (GP mean / UCB / std per action type)
        4. 3-agent pipeline (Planner → Hypothesis → Scorer):
           a. Planner takes (pk, yk) — re-ranks principles with EXPLORE/VALIDATE/REFINE actions
           b. Hypothesis generates directional hypothesis from action-annotated principles
           c. Scorer re-ranks top-k candidates; hybrid score = acq + LLM alignment
        5. Greedy selection → oracle evaluation → principle extraction

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

# wandb is optional — gracefully disabled when not installed or not configured
try:
    import wandb as _wandb_module
    _WANDB_AVAILABLE = True
except ImportError:
    _wandb_module = None
    _WANDB_AVAILABLE = False

# ── path setup ────────────────────────────────────────────────────────────────
# Both gollum and pigollum packages live under src/
_HERE = Path(__file__).parent.resolve()
_SRC = _HERE / "src"
if _SRC.is_dir():
    sys.path.insert(0, str(_SRC))

from gollum.data.module import BaseDataModule
from gollum.utils.config import instantiate_class
from gollum.data.utils import torch_delete_rows
from botorch.utils.multi_objective.pareto import is_non_dominated

from pigollum.bo.pi_optimizer import PiGollumOptimizer
from pigollum.principle.buffer import PrincipleBuffer
from pigollum.principle.extractor import PrincipleExtractor
from pigollum.principle.scorer import PrincipleScorer
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

    device_str = surr_cfg.get("init_args", {}).get("device", "cpu")

    scorer = PrincipleScorer(
        embedding_model_name=embedding_model,
        lambda_factor=pi_cfg.get("lambda_factor", 0.5),
        plateau_threshold=pi_cfg.get("plateau_threshold", 0.1),
        plateau_count=pi_cfg.get("plateau_count", 3),
        device=device_str,
    )

    batch_size = bo_cfg.get("init_args", {}).get("batch_size", 1)

    optimizer = PiGollumOptimizer(
        surrogate_model_config=surr_cfg,
        acq_function_config=acq_cfg,
        batch_size=batch_size,
        principle_weight=principle_weight,
        min_principles_for_guidance=min_principles,
        principle_weight_schedule=weight_schedule,
        extractor=extractor,
        scorer=scorer,
        enable_post_acq_agents=pi_cfg.get("enable_post_acq_agents", False),
        top_k_for_rescoring=pi_cfg.get("top_k_for_rescoring", 20),
        include_experimental_data=pi_cfg.get("include_experimental_data", True),
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

    multi_obj = len(objective_names) > 1
    if multi_obj:
        dominated_mask = is_non_dominated(torch.tensor(train_y_np))
        pareto_size = int(dominated_mask.sum().item())
    else:
        pareto_size = None

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
        bsf = best_so_far[i] if hasattr(best_so_far, "__len__") else float(best_so_far)
        print(f"  {obj}: new={new_vals[0]:.4f}  best_so_far={bsf:.4f}")
    if multi_obj:
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
        "best_so_far": best_so_far.tolist() if hasattr(best_so_far, "tolist") else float(best_so_far),
        "pareto_size": pareto_size,
        "n_principles": principle_buffer.size,
        "n_principles_by_source": {
            "broad": n_broad, "refined": n_refined,
            "gp": n_gp, "oracle": n_oracle,
        },
    })


# ─────────────────────────────────────────────────────────────────────────────
# W&B helpers
# ─────────────────────────────────────────────────────────────────────────────

def _wandb_active() -> bool:
    return _WANDB_AVAILABLE and _wandb_module.run is not None


def wandb_log_iteration(
    iteration: int,
    optimizer: "PiGollumOptimizer",
    journal: "PrincipleJournal",
    objective_names: List[str],
    selected_outcomes: List[Dict],
    best_train_y: List[float],
) -> None:
    """Log per-iteration metrics, principle table, and agent outputs to W&B."""
    if not _wandb_active():
        return

    log_dict: Dict = {"iteration": iteration}

    # ── BO objectives ─────────────────────────────────────────────────
    for name, val in zip(objective_names, best_train_y):
        log_dict[f"best_so_far/{name}"] = val
    for out in selected_outcomes:
        for name, val in out.items():
            log_dict[f"new_candidate/{name}"] = val

    action = optimizer._last_action_info.get("action_type", "unknown")
    log_dict["action_type"] = action
    log_dict["n_principles"] = optimizer.principle_buffer.size

    # ── Directional hypothesis (full structured text) ─────────────────
    if optimizer._last_direction_hyp:
        log_dict["direction_hypothesis"] = optimizer._last_direction_hyp
    if optimizer._last_planner_response:
        log_dict["planner_response"] = optimizer._last_planner_response
    if optimizer._last_scorer_response:
        log_dict["scorer_response"] = optimizer._last_scorer_response

    # ── Principle scores table ────────────────────────────────────────
    if journal._records:
        last_rec = journal._records[-1]
        if last_rec.principle_scores:
            table = _wandb_module.Table(columns=[
                "principle_id", "source", "iteration_added",
                "reward", "exploration_score", "exploitation_score",
                "final_score", "selected", "principle_text",
            ])
            for snap in last_rec.principle_scores:
                table.add_data(
                    snap.principle_id[:8],
                    snap.source,
                    snap.iteration_added,
                    round(snap.reward, 4),
                    round(snap.exploration_score, 4),
                    round(snap.exploitation_score, 4),
                    round(snap.final_score, 4),
                    snap.was_selected,
                    snap.principle_text[:120],
                )
            log_dict[f"principles/iter_{iteration:03d}"] = table

            # Also log per-principle scalars for easy charting
            for snap in last_rec.principle_scores:
                pid = snap.principle_id[:8]
                log_dict[f"principle_scores/reward/{pid}"] = snap.reward
                log_dict[f"principle_scores/exploration/{pid}"] = snap.exploration_score
                log_dict[f"principle_scores/exploitation/{pid}"] = snap.exploitation_score
                log_dict[f"principle_scores/final/{pid}"] = snap.final_score

    _wandb_module.log(log_dict, step=iteration)


# ─────────────────────────────────────────────────────────────────────────────
# Main experiment loop
# ─────────────────────────────────────────────────────────────────────────────

def run(cfg: dict, data_root: str, output_dir: str, seed: int) -> None:
    torch.manual_seed(seed)
    np.random.seed(seed)

    os.makedirs(output_dir, exist_ok=True)

    # ── W&B initialisation ───────────────────────────────────────────────────
    wandb_cfg = cfg.get("wandb", {})
    if _WANDB_AVAILABLE and wandb_cfg.get("enabled", False):
        _wandb_module.init(
            project=wandb_cfg.get("project", "pigollum"),
            name=wandb_cfg.get("run_name", None),
            tags=wandb_cfg.get("tags", []),
            config=cfg,
            dir=output_dir,
        )
        logger.info("W&B run initialised: %s", _wandb_module.run.name)
    elif wandb_cfg.get("enabled", False) and not _WANDB_AVAILABLE:
        logger.warning("wandb not installed — W&B logging disabled. Run: pip install wandb")

    # ------------------------------------------------------------------
    # 1. Data module
    # ------------------------------------------------------------------
    logger.info("Loading dataset…")
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
    task_context = pi_cfg.get("task_context", "")

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
    # 6. Main BO loop
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

        # --- W&B logging ---
        wandb_log_iteration(
            iteration=iteration,
            optimizer=optimizer,
            journal=journal,
            objective_names=objective_names,
            selected_outcomes=selected_outcomes,
            best_train_y=train_y.cpu().numpy().max(axis=0).tolist(),
        )

        # --- Flush results to disk after every iteration ---
        with open(os.path.join(output_dir, "bo_results.json"), "w") as f:
            json.dump(results, f, indent=2)
        optimizer.principle_buffer.save(os.path.join(output_dir, "principles.json"))
        journal.save(os.path.join(output_dir, "journal.json"), buffer=optimizer.principle_buffer)
        journal.save_text_report_iterations(os.path.join(output_dir, "evolution_report.txt"))

    # ------------------------------------------------------------------
    # 7. Save final outputs (report requires full buffer)
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

    if _wandb_active():
        _wandb_module.log({
            f"final_best/{name}": float(val)
            for name, val in zip(objective_names, best)
        })
        _wandb_module.finish()
        logger.info("W&B run finished.")


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
        "--data_root", default="../gollum",
        help="Root directory containing the data/ folder (e.g. ../gollum/)",
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
    parser.add_argument(
        "--wandb_project", default=None,
        help="Enable W&B logging and set the project name (overrides config)",
    )
    parser.add_argument(
        "--wandb_run_name", default=None,
        help="W&B run name (overrides config)",
    )
    args = parser.parse_args()

    cfg = load_config(args.config)
    if args.n_iters is not None:
        cfg["n_iters"] = args.n_iters
    if args.wandb_project is not None:
        cfg.setdefault("wandb", {})
        cfg["wandb"]["enabled"] = True
        cfg["wandb"]["project"] = args.wandb_project
    if args.wandb_run_name is not None:
        cfg.setdefault("wandb", {})
        cfg["wandb"]["run_name"] = args.wandb_run_name

    run(
        cfg=cfg,
        data_root=args.data_root,
        output_dir=args.output_dir,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
