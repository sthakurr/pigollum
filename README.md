**PiGollum — Architecture & Implementation Plan**

Core Idea
PiFlow extracts scientific principles from (hypothesis, experiment) pairs and uses exploration-exploitation scoring over those principles to guide search. Gollum runs LLM-featurized Bayesian Optimization with GP surrogates. PiGollum fuses both: after every BO oracle evaluation, an LLM extracts a principle from the observed (sequence, outcome) pair; these principles then re-rank BO acquisition scores at the next iteration.

Integration Mechanism

BO acqf score (ESM-2 GP)   ×(1−α)
                              ╲
                                ⟹  final_score  →  greedy top-k
                              ╱
Principle score (PiFlow)   ×α
The principle score = λ · semantic_diversity + (1−λ) · reward_exploitation (exact PiFlow algorithm), computed via cosine similarity between sentence-transformer embeddings of candidate descriptions and the buffer of discovered principles.

Files Built (pigollum/)
File	Purpose
src/pigollum/principle/buffer.py	Principle dataclass + PrincipleBuffer store with save/load
src/pigollum/principle/extractor.py	LLM-based hypothesis + principle extraction; statistical fallback if no LLM
src/pigollum/principle/scorer.py	PiFlow's exact exploration-exploitation scoring; candidate re-ranking
src/pigollum/bo/pi_optimizer.py	PiGollumOptimizer extends BotorchOptimizer with principle guidance
src/pigollum/utils/llm_client.py	OpenAI-compatible LLM client with graceful fallback
src/pigollum/utils/sequence_utils.py	Amino acid sequence → text description for LLM prompts
run_experiment.py	Full biocat BO loop with principle warm-start and per-iteration logging
configs/biocat_pigollum.yaml	Complete biocat config (ESM-2 + DualDeepGP + PiGollum settings)
How to Run
Without LLM (statistical principles, still fully functional):


conda run -n gollum python run_experiment.py \
  --config configs/biocat_pigollum.yaml \
  --data_root gollum-2 \
  --output_dir results/biocat_pigollum \
  --n_iters 10
  
With LLM (GPT-4o-mini or local Ollama):


export PIGOLLUM_LLM_API_KEY=sk-...
export PIGOLLUM_LLM_MODEL=gpt-4o-mini
conda run -n gollum python run_experiment.py --config configs/biocat_pigollum.yaml
Key Config Knobs (pigollum: section)
principle_weight (α=0.3) — how much to trust principles vs BO
principle_weight_schedule: linear — α grows as more principles accumulate
lambda_factor (0.5) — exploration vs exploitation within principle scoring
warm_start_principles: true — extract principles from initial training set before the first BO iteration
min_principles_for_guidance: 3 — mirrors PiFlow's bootstrap threshold