# PiGollum

**Principle-Guided Bayesian Optimisation with GP as Experiment Agent**

> Fusing [PiFlow](https://arxiv.org/abs/2505.15047) principle-aware reasoning with [Gollum](https://github.com/schwallergroup/gollum) LLM-featurised BO for principled scientific discovery.

---

## Overview

PiGollum combines two complementary ideas:

| Component | Role |
|-----------|------|
| **Gollum** | Bayesian Optimisation with ESM-2–featurised DualDeepGP surrogate and multi-objective acquisition |
| **PiFlow** | Iterative extraction of scientific principles from (hypothesis, experiment) pairs, with exploration-exploitation scoring to guide search |

After every oracle evaluation, an LLM extracts a generalizable principle from the observed `(sequence → outcome)` pair. These principles accumulate in a buffer and re-rank the BO acquisition scores at the next iteration — gradually shifting the search from statistical optimality toward *mechanistically sound* regions of sequence space.

A **post-acquisition 3-agent pipeline** (Planner → Hypothesis → Scorer) further refines candidate selection after the GP scores are computed, ensuring the final pick is both statistically strong and scientifically coherent.

---

## Architecture

```
                        ┌──────────────────────────────────────────────────┐
  WARM-START (once)     │  LLM → 5 broad principles                       │
                        │  Evidence retrieval → LLM refines each principle │
                        └──────────────────────────────────────────────────┘

  PER BO ITERATION
  ─────────────────────────────────────────────────────────────────────────
  1. Train DeepGP on (train_x, train_y)
  2. GP predicts means + stds for all candidates   ← Experiment Agent

  3. Inner PiFlow loop  (n_inner_steps × per iteration)
     ┌─ Scorer  → action: explore / refine / validate
     ├─ Planner → select candidates + build guidance
     ├─ LLM     → predictive hypothesis per candidate   ← Hypothesis Agent
     ├─ GP      → validate hypothesis (posterior)       ← Experiment Agent
     └─ LLM     → extract principle from (hyp, GP pred) ← Principle Agent

  4. Score all candidates
     final = (1−α) × GP_score  +  α × principle_alignment

  5. Post-acquisition 3-agent pipeline
     ┌─ Agent 1 · Planner    → re-rank principles given new evidence
     ├─ Agent 2 · Hypothesis → directional hypothesis for next candidate
     └─ Agent 3 · Scorer     → re-rank top-k by hypothesis alignment

  6. Greedy select → Oracle evaluation → extract oracle principle
  ─────────────────────────────────────────────────────────────────────────
```

---

## Key Features

- **GP as Experiment Agent** — the GP posterior validates LLM hypotheses in the inner loop, replacing expensive wet-lab experiments with in-silico validation.
- **Principle buffer** — discovered principles are stored with embeddings, rewards, and provenance (`broad` / `refined` / `gp` / `oracle`).
- **PiFlow Min-Max action selection** — exploration-exploitation scoring drives `explore` / `refine` / `validate` decisions per iteration.
- **Post-acquisition 3-agent pipeline** — Planner, Hypothesis, and Scorer agents inject scientific reasoning between the acquisition function and greedy selection.
- **Three LLM backends** — local HuggingFace, any OpenAI-compatible API, or Google Gemini (no extra SDK required).
- **W&B integration** — logs BO metrics, per-principle scores, and full agent outputs at every iteration.

---

## Installation

### Prerequisites

- CUDA-capable GPU (A6000 or equivalent recommended for local LLM inference)
- [Conda](https://docs.conda.io/en/latest/miniconda.html) / [Mamba](https://github.com/mamba-org/mamba)
- The `gollum` repository cloned as a sibling directory (`~/gollum/`)

### Create the environment

```bash
# Clone the repo
git clone <repo-url> pigollum
cd pigollum

# Create and activate conda environment
conda env create -f requirements.yaml
conda activate pigollum
```

### Data

Place the biocat dataset under `~/gollum/data/biocat/`:

```
~/gollum/
└── data/
    └── biocat/
        ├── enzymes_sequence_yield_ee_processed_ddg_yeo-johnson.csv   # 200 labelled
        └── enzymes_sequence_10k.csv                                   # 10k candidates
```

---

## Quick Start

### Local HuggingFace model (default)

```bash
conda activate pigollum
python run_experiment.py \
  --config configs/biocat_pigollum.yaml \
  --data_root ../gollum \
  --output_dir results/biocat_pigollum \
  --n_iters 10
```

### OpenAI-compatible API (GPT-4o-mini, Together, Groq, Ollama, …)

```bash
export PIGOLLUM_LLM_API_KEY=sk-...
export PIGOLLUM_LLM_MODEL=gpt-4o-mini          # or llama-3-70b, etc.
# export PIGOLLUM_LLM_BASE_URL=http://localhost:11434/v1  # for local servers

python run_experiment.py --config configs/biocat_pigollum.yaml
```

### Google Gemini

```bash
export GEMINI_API_KEY=AIza...
# model defaults to gemini-2.0-flash; override with GEMINI_MODEL=gemini-1.5-pro
```

Then set `backend: gemini` in `configs/biocat_pigollum.yaml`.

### With W&B tracking

```bash
python run_experiment.py \
  --config configs/biocat_pigollum.yaml \
  --wandb_project my-enzyme-discovery \
  --wandb_run_name biocat-run-01
```

---

## Configuration

All settings live under the `pigollum:` key in the YAML config.

### Core BO settings

| Key | Default | Description |
|-----|---------|-------------|
| `principle_weight` | `0.3` | α — weight of principle scores vs GP/acquisition scores |
| `principle_weight_schedule` | `linear` | How α grows: `constant` / `linear` / `step` |
| `min_principles_for_guidance` | `3` | Minimum buffer size before principle guidance activates |
| `n_inner_steps` | `5` | Hypothesis-validation cycles per BO iteration |
| `candidate_sample_size` | `10` | Candidates considered per inner step |
| `lambda_factor` | `0.5` | λ — exploration vs exploitation within principle scoring |
| `warm_start_principles` | `true` | Run broad → refined warm-start before iteration 1 |
| `n_broad_principles` | `5` | Number of domain-knowledge seed principles |

### Post-acquisition 3-agent pipeline

| Key | Default | Description |
|-----|---------|-------------|
| `enable_post_acq_agents` | `true` | Enable Planner → Hypothesis → Scorer after acquisition scoring |
| `top_k_for_rescoring` | `20` | How many top-k candidates the Scorer agent re-ranks |
| `include_experimental_data` | `true` | Pass oracle history to the Hypothesis agent |

### LLM backend

| Key | Default | Description |
|-----|---------|-------------|
| `backend` | `hf` | `hf` / `api` / `gemini` |
| `hf_model_name` | `Qwen/Qwen2.5-7B-Instruct` | HuggingFace model ID (local inference) |
| `hf_torch_dtype` | `bfloat16` | Model weight precision |
| `llm_api_key` | `null` | API key (or set `PIGOLLUM_LLM_API_KEY`) |
| `llm_base_url` | `null` | Base URL for OpenAI-compatible servers |
| `llm_model` | `null` | Model name for API / Gemini backends |

### W&B logging

| Key | Default | Description |
|-----|---------|-------------|
| `wandb.enabled` | `false` | Enable W&B logging (or use `--wandb_project` CLI flag) |
| `wandb.project` | `pigollum` | W&B project name |
| `wandb.run_name` | `null` | Run name (auto-generated if null) |
| `wandb.tags` | `[]` | Tags attached to the run |

---

## Outputs

All outputs are written to `--output_dir` (default: `results/biocat_pigollum/`).

| File | Contents |
|------|----------|
| `bo_results.json` | Per-iteration: selected sequences, observed objectives, best-so-far, Pareto front size, principle counts by source |
| `principles.json` | Full principle buffer — text, hypothesis, embedding, reward, source, GP confidence |
| `journal.json` | Complete lineage: per-iteration action type, all principle scores (exploration / exploitation / final), selected candidates, winning principle analysis |
| `evolution_report.txt` | Human-readable timeline; top-5 principles by reward, selection count, and influence score |

### W&B logged metrics (per iteration)

| Metric | Description |
|--------|-------------|
| `best_so_far/{objective}` | Best observed value for each objective |
| `new_candidate/{objective}` | Value of the newly evaluated candidate |
| `action_type` | `explore` / `refine` / `validate` |
| `n_principles` | Total principles in buffer |
| `direction_hypothesis` | Full structured output of the Hypothesis agent |
| `planner_response` | Full structured output of the Planner agent |
| `scorer_response` | Full structured output of the Scorer agent |
| `principles/iter_NNN` | Table of all principles with scores at iteration N |
| `principle_scores/{type}/{id}` | Per-principle reward / exploration / exploitation / final scalars |

---

## Project Structure

```
pigollum/
├── configs/
│   ├── biocat_pigollum.yaml      # Main experiment config
│   ├── bh_pigollum.yaml          # Placeholder config
│   └── flip2_pigollum.yaml       # Placeholder config
│
├── src/pigollum/
│   ├── bo/
│   │   └── pi_optimizer.py       # PiGollumOptimizer: GP training, inner loop,
│   │                             #   post-acquisition 3-agent pipeline, selection
│   ├── principle/
│   │   ├── buffer.py             # Principle dataclass + in-memory buffer
│   │   ├── extractor.py          # LLM backends + all agent prompts and methods
│   │   ├── scorer.py             # Exploration-exploitation scoring; candidate ranking
│   │   ├── planner.py            # Inner loop orchestration; guidance synthesis
│   │   └── journal.py            # Iteration snapshots; evolution report generation
│   └── utils/
│       ├── llm_client.py         # OpenAI-compatible + Gemini client builders
│       └── sequence_utils.py     # Amino acid composition → text description
│
├── run_experiment.py             # Experiment entry point
├── setup.py                      # Package metadata
└── requirements.yaml             # Conda environment
```

---

## LLM Agent Prompts

The three post-acquisition agents follow the structured format from [PiFlow Appendix Q](https://arxiv.org/abs/2505.15047):

| Agent | Output Format | Purpose |
|-------|--------------|---------|
| **Planner** | Understand Evidence → Clarify GAP → Connect to Principle → Principle Statement → Re-ranked Indices → Double-check | Re-ranks principle buffer after each BO iteration |
| **Hypothesis** | Rationale (Major + Minor premise) → Hypothesis → Reiterate → Ideal Candidate Profile | Generates a directional hypothesis for the next candidate |
| **Scorer** | `Candidate N: score` (0–10 per candidate) | Rates top-k candidates by scientific alignment with the hypothesis |

---

## Acknowledgements

- [PiFlow](https://arxiv.org/abs/2505.15047) — Principle-Aware Scientific Discovery with Multi-Agent Collaboration (Pu et al., 2026)
- [Gollum](https://github.com/schwallergroup/gollum) — LLM-guided Bayesian Optimisation for molecular design
- [BoTorch](https://botorch.org/) — Bayesian Optimisation in PyTorch
- [ESM-2](https://github.com/facebookresearch/esm) — Evolutionary Scale Modeling for protein sequences
