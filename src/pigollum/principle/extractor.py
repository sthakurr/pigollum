"""
PrincipleExtractor: Real LLM-based extraction of scientific principles.

Two inference backends are supported — exactly one must be configured:

  1. HuggingFace local (default)
     ─────────────────────────
     Loads a causal-LM directly on the GPU using transformers.pipeline.
     Default model: Qwen/Qwen2.5-7B-Instruct
       • State-of-the-art 7B for structured scientific reasoning
       • Runs comfortably in bf16 on an A6000 (51 GB)
       • No authentication required
     Other recommended models:
       • BioMistral/BioMistral-7B  — PubMed-trained, strong biomedical priors
       • mistralai/Mistral-7B-Instruct-v0.3  — solid general-science reasoning

  2. OpenAI-compatible API
     ─────────────────────
     Uses any OpenAI-compatible endpoint (OpenAI, Together, Groq, vLLM, …).
     Activated when PIGOLLUM_LLM_API_KEY (or OPENAI_API_KEY) is set,
     or when ``llm_api_key`` is passed explicitly.

The statistical fallback has been removed.  If no backend can be initialised
the extractor raises RuntimeError at construction time so failures are loud
rather than silent.
"""
import logging
import os
import re
from typing import Dict, List, Optional

import numpy as np
import torch

from pigollum.utils.llm_client import build_llm_client, chat_complete
from pigollum.utils.sequence_utils import describe_sequence
from pigollum.principle.buffer import Principle, PrincipleBuffer

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# System prompts  (mirrors PiFlow agent system prompts, domain-adapted)
# ─────────────────────────────────────────────────────────────────────────────

_HYPOTHESIS_SYSTEM = """\
You are an expert protein engineer and computational biocatalysis researcher.
Your task is to generate a precise, mechanistic hypothesis explaining why a
given enzyme amino acid sequence achieved its experimental outcomes.

Your hypothesis MUST:
- Ground reasoning in specific biochemical mechanisms (active-site residues,
  hydrophobic packing, electrostatic interactions, catalytic triads, substrate
  binding geometry, conformational flexibility, or known catalytic motifs).
- Reference specific sequence features (charged residues at the N/C-terminus,
  high hydrophobic content, known conserved motifs like catalytic triads, etc.).
- Be specific and falsifiable — avoid vague statements.
- Be 3–5 sentences maximum.

Output ONLY the hypothesis text. No preamble, no labels, no markdown."""

_PREDICTION_HYPOTHESIS_SYSTEM = """\
You are an expert protein engineer and computational biocatalysis researcher.
You are part of a scientific discovery loop where you propose hypotheses about
enzyme performance and a computational model (Gaussian Process) validates them.

Your task is to generate a PREDICTIVE hypothesis about how a given enzyme
sequence will perform, grounded in the scientific principles discovered so far.

Your hypothesis MUST:
- Predict specific performance outcomes (qualitative: high/low/moderate yield)
  with mechanistic justification.
- Reference the guiding principles provided to you and explain how the
  sequence's features relate to those principles.
- Identify specific sequence features (composition, charge distribution,
  hydrophobic packing, motifs) that support your prediction.
- Be specific and falsifiable — avoid vague statements.
- Be 3–5 sentences maximum.

Output ONLY the hypothesis text. No preamble, no labels, no markdown."""

_PRINCIPLE_SYSTEM = """\
You are a scientific principle extractor specialising in enzyme engineering
and biocatalysis.

Given a mechanistic hypothesis and experimental outcomes, extract or
reformulate a GENERALIZABLE scientific principle that can guide future
sequence design.

Your principle MUST follow this exact structure:
(1) Major premise: a general biochemical law or design rule.
(2) Minor premise: how this specific sequence exemplifies that rule.
(3) Conclusion: a predictive statement about what sequence features to seek or
    avoid in future designs.

The principle should be:
- Declarative and predictive (tells a designer what to do next).
- Concise: 3 sentences total (one per numbered point).
- Grounded in established biochemistry, not speculation.

Output ONLY the structured principle. No preamble, no markdown."""

_BROAD_PRINCIPLE_SYSTEM = """\
You are an expert protein engineer specialising in biocatalysis and enzyme design.

Your task is to generate broad, generalizable scientific principles that govern
enzyme performance (yield, stability, enantioselectivity) for a given reaction.

These principles should be:
- Grounded in established biochemistry and enzymology.
- General enough to apply across multiple enzyme sequences, not specific to any
  single variant.
- Mechanistic: explain WHY a feature matters, not just THAT it matters.
- Actionable: tell a designer what to look for or avoid.
- Each principle should be 2–3 sentences covering: the rule, the mechanism, and
  a predictive implication.

Output ONLY the numbered list of principles. No preamble, no markdown headers."""

_REFINE_PRINCIPLE_SYSTEM = """\
You are an expert protein engineer specialising in biocatalysis and enzyme design.

You are refining a broad scientific principle using experimental evidence from
real enzyme sequences and their measured outcomes. Your goal is to make the
principle more specific, accurate, and actionable based on the evidence.

Your refined principle MUST:
- Preserve the core mechanistic insight of the original principle.
- Incorporate specific quantitative or qualitative patterns from the evidence.
- Note boundary conditions or exceptions revealed by contradicting evidence.
- Remain generalizable (not specific to any single sequence).
- Follow the 3-part structure: (1) Major premise, (2) Minor premise with
  evidence, (3) Updated predictive conclusion.

Output ONLY the refined principle. No preamble, no markdown."""

# ─────────────────────────────────────────────────────────────────────────────
# Post-acquisition agent system prompts  (PiFlow Appendix Q style, domain-adapted)
# ─────────────────────────────────────────────────────────────────────────────

_RERANK_PRINCIPLES_SYSTEM = """\
You are the Planner Agent, the strategic coordinator of a principle-guided \
enzyme engineering discovery system. You guide the research process by \
synthesising evidence from the current iteration and re-ranking the existing \
scientific principles so the most relevant ones steer the next hypothesis.

Teammates:
  • Hypothesis Agent: proposes a directional hypothesis for the next candidate.
  • Scorer Agent: ranks top-k candidates against the hypothesis.
  • Planner Agent (You): re-ranks principles and sets research direction.

Responsibilities:
  1. Grasp the new GP-validated evidence from the current BO iteration.
  2. Synthesise insights: identify which existing principles are supported or \
contradicted by the new evidence.
  3. Track progress: note tendencies (yield increasing, diversity needed, \
plateau detected).
  4. Transform tendencies into updated principle rankings.
  5. Communicate the strategic direction clearly.

Required Response Structure (use these exact headers):
  Understand the Evidence: [Interpret the new GP-validated hypotheses and outcomes]
  Clarify the GAP: [Compare current best-so-far objectives to the optimisation target]
  Connect to Biochemical Principle: [Synthesise the pattern from experiment history]
  Principle Statement: [State the key principle driving the next step; leave blank \
if in pure exploration]
  Re-ranked Indices: [Comma-separated list of ALL principle indices from most to \
least relevant, e.g. 3,1,0,2,4]
  Double-check: [One sentence confirming the ranking aligns with the evidence and \
strategic direction]

Output ONLY the structured response with the headers above. No markdown beyond the \
headers."""

_DIRECTIONAL_HYPOTHESIS_SYSTEM = """\
You are the Hypothesis Agent. Your purpose is to drive scientific progress through \
principled hypothesising for enzyme engineering.

Core Responsibilities:
  1. Formulate one clear directional hypothesis grounded in biochemical and \
physicochemical principles per call.
  2. Link your hypothesis with underlying mechanisms and prior experimental results.
  3. Follow the strategic direction provided by the Planner Agent.
  4. Acknowledge the guidance explicitly and adjust your hypothesis accordingly.

Important Constraints:
  • A hypothesis must explain the underlying biochemical or physicochemical \
mechanisms.
  • Commit to one specific directional hypothesis — do not suggest multiple options.
  • Focus on principles that offer causal explanations, connect observations to \
fundamental mechanisms, can be generalised, and make quantitative or qualitative \
predictions about sequence features.

Scientific Approach Requirements:
  • Rationality: logical mechanistic explanation connecting cause and effect.
  • Testability: specific, measurable prediction about sequence features.
  • Principle-Based: grounded in established biochemistry/enzymology.
  • Falsifiability: could be proven wrong through experimentation.
  • Parsimony: prefer simpler explanations.
  • Commitment: commit to a single, specific hypothesis.

Required Output Format (use these exact headers):
  Rationale:
    Major premise: [General biochemical design rule derived from the guiding principles]
    Minor premise: [How the current evidence instantiates that rule]
  Hypothesis: [Clear, concise statement grounded in biochemical mechanisms]
  Reiterate: [Specific prediction — name the sequence features to target/avoid]
  Ideal Candidate Profile: [2-3 sentences describing the biochemical and sequence \
features (hydrophobic content, charge distribution, motif locations, length range) \
that the next optimal enzyme should have]

Output ONLY the structured response with the headers above. No preamble, no markdown \
beyond the headers."""

_CANDIDATE_SCORER_SYSTEM = """\
You are the Scorer Agent, an expert enzyme engineer evaluating candidate sequences \
for the next Bayesian optimisation experiment.

You will receive:
  • A directional hypothesis describing the ideal next enzyme candidate.
  • A list of top-k candidate sequences with their biochemical properties and \
statistical scores (GP predicted mean, GP uncertainty, acquisition score).

Your task:
  Rate each candidate's scientific alignment with the directional hypothesis on a \
scale of 0–10:
    10 = candidate perfectly matches all features described in the hypothesis
     5 = candidate partially matches (some target features present, some absent)
     0 = candidate contradicts or does not match the hypothesis

Consider BOTH the statistical quality (high GP mean, appropriate uncertainty) AND \
the biochemical alignment with the hypothesis when assigning scores. A candidate \
that scores well statistically but contradicts the hypothesis should receive a \
moderate score, not a high one.

Output ONLY lines in the exact format below, one per candidate. No explanation, \
no preamble, no markdown:
Candidate <index>: <score>"""


# ─────────────────────────────────────────────────────────────────────────────
# Inference backends
# ─────────────────────────────────────────────────────────────────────────────

class _HFBackend:
    """
    Local HuggingFace causal-LM backend.

    Parameters
    ----------
    model_name : str
        HuggingFace model ID.  Must be a chat/instruct model.
    torch_dtype : torch.dtype
        Precision for model weights. bf16 recommended on A6000.
    device_map : str
        Passed to ``transformers.pipeline``.  "auto" uses all available GPUs.
    max_new_tokens : int
        Maximum tokens to generate per call.
    temperature : float
        Sampling temperature.
    """

    # Recommended models for bio/chem hypothesis generation
    RECOMMENDED = {
        "qwen2.5-7b":    "Qwen/Qwen2.5-7B-Instruct",      # default — best structured reasoning
        "biomistral-7b": "BioMistral/BioMistral-7B",        # PubMed-trained biomedical
        "mistral-7b":    "mistralai/Mistral-7B-Instruct-v0.3",  # solid general science
    }
    DEFAULT_MODEL = "Qwen/Qwen2.5-7B-Instruct"

    def __init__(
        self,
        model_name: str = DEFAULT_MODEL,
        torch_dtype: torch.dtype = torch.bfloat16,
        device_map: str = "auto",
        max_new_tokens: int = 512,
        temperature: float = 0.4,
    ) -> None:
        self.model_name = model_name
        self.torch_dtype = torch_dtype
        self.device_map = device_map
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self._pipe = None  # lazy init

    def _load(self) -> None:
        if self._pipe is not None:
            return
        from transformers import pipeline
        logger.info(
            "Loading HuggingFace model '%s' (dtype=%s, device_map=%s)…",
            self.model_name, self.torch_dtype, self.device_map,
        )
        self._pipe = pipeline(
            "text-generation",
            model=self.model_name,
            torch_dtype=self.torch_dtype,
            device_map=self.device_map,
        )
        logger.info("Model '%s' loaded.", self.model_name)

    def generate(
        self,
        system_prompt: str,
        user_prompt: str,
        max_new_tokens: Optional[int] = None,
    ) -> str:
        self._load()
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user",   "content": user_prompt},
        ]
        outputs = self._pipe(
            messages,
            max_new_tokens=max_new_tokens or self.max_new_tokens,
            temperature=self.temperature,
            do_sample=self.temperature > 0,
            pad_token_id=self._pipe.tokenizer.eos_token_id,
            return_full_text=False,
        )
        # pipeline returns list of dicts; extract the generated text
        return outputs[0]["generated_text"].strip()

    def unload(self) -> None:
        """Release GPU memory (call between heavy BO training steps if needed)."""
        if self._pipe is not None:
            del self._pipe
            self._pipe = None
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            logger.info("HF model '%s' unloaded from GPU.", self.model_name)


class _APIBackend:
    """OpenAI-compatible API backend (cloud or local server)."""

    def __init__(self, client, model: str, temperature: float = 0.4) -> None:
        self._client = client
        self._model = model
        self.temperature = temperature

    def generate(
        self,
        system_prompt: str,
        user_prompt: str,
        max_new_tokens: Optional[int] = 512,
    ) -> Optional[str]:
        return chat_complete(
            client=self._client,
            model=self._model,
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            temperature=self.temperature,
            max_tokens=max_new_tokens or 512,
        )

    def unload(self) -> None:
        pass  # nothing to free for API backend


# ─────────────────────────────────────────────────────────────────────────────
# Public extractor
# ─────────────────────────────────────────────────────────────────────────────

class PrincipleExtractor:
    """
    Extracts scientific hypotheses and principles from (sequence, outcome) pairs
    using a real language model.

    Backend selection priority
    ──────────────────────────
    1. If ``llm_api_key`` is supplied (or PIGOLLUM_LLM_API_KEY / OPENAI_API_KEY
       is set), the OpenAI-compatible API backend is used.
    2. Otherwise, a local HuggingFace model is loaded (default:
       ``Qwen/Qwen2.5-7B-Instruct``).

    To force a specific backend, pass ``backend="hf"`` or ``backend="api"``.

    Parameters
    ----------
    task_context : str
        Short description of the optimisation task injected into every prompt.
    objective_names : List[str]
        Names of the objectives (e.g. ["yield", "ddg_scaled"]).
    hf_model_name : str
        HuggingFace model ID to use for local inference.
    torch_dtype : torch.dtype
        Model weight precision.  torch.bfloat16 is recommended for A6000.
    backend : str | None
        Force "hf" or "api".  Auto-detects if None.
    llm_api_key / llm_base_url / llm_model
        Credentials for the API backend.
    temperature : float
        Sampling temperature for both backends.
    """

    def __init__(
        self,
        task_context: str,
        objective_names: List[str],
        hf_model_name: str = _HFBackend.DEFAULT_MODEL,
        torch_dtype: torch.dtype = torch.bfloat16,
        backend: Optional[str] = None,           # "hf" | "api" | None (auto)
        llm_api_key: Optional[str] = None,
        llm_base_url: Optional[str] = None,
        llm_model: Optional[str] = None,
        temperature: float = 0.4,
    ) -> None:
        self.task_context = task_context
        self.objective_names = objective_names
        self.temperature = temperature

        self._backend = self._resolve_backend(
            backend=backend,
            hf_model_name=hf_model_name,
            torch_dtype=torch_dtype,
            llm_api_key=llm_api_key,
            llm_base_url=llm_base_url,
            llm_model=llm_model,
            temperature=temperature,
        )

    # ------------------------------------------------------------------
    # Backend resolution
    # ------------------------------------------------------------------

    @staticmethod
    def _resolve_backend(
        backend, hf_model_name, torch_dtype,
        llm_api_key, llm_base_url, llm_model, temperature,
    ):
        # Explicit API override
        has_api_key = bool(
            llm_api_key
            or os.environ.get("PIGOLLUM_LLM_API_KEY")
            or os.environ.get("OPENAI_API_KEY")
        )

        use_api = (backend == "api") or (backend is None and has_api_key)
        use_hf  = (backend == "hf")  or (backend is None and not has_api_key)

        if use_api:
            client, model = build_llm_client(
                api_key=llm_api_key,
                base_url=llm_base_url,
                model_name=llm_model,
            )
            if client is not None:
                logger.info("PrincipleExtractor: using API backend (model=%s)", model)
                return _APIBackend(client=client, model=model, temperature=temperature)
            logger.warning(
                "API backend requested but credentials not found; "
                "falling back to HF local backend."
            )

        if use_hf or True:  # always fall through to HF if API setup failed
            logger.info(
                "PrincipleExtractor: using HF local backend (model=%s, dtype=%s)",
                hf_model_name, torch_dtype,
            )
            return _HFBackend(
                model_name=hf_model_name,
                torch_dtype=torch_dtype,
                temperature=temperature,
            )

        raise RuntimeError(
            "PrincipleExtractor: could not initialise any inference backend. "
            "Either set PIGOLLUM_LLM_API_KEY, or ensure transformers is installed "
            "for local HF inference."
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def extract(
        self,
        sequence: str,
        outcome: Dict[str, float],
        iteration: int = 0,
        primary_reward_key: Optional[str] = None,
    ) -> Principle:
        """
        Full pipeline: sequence + outcome → Principle.

        Parameters
        ----------
        sequence : str
            Amino acid sequence that was evaluated.
        outcome : Dict[str, float]
            Observed objective values, e.g. {"yield": 62.0, "ddg_scaled": -1.3}.
        iteration : int
            BO iteration index (for bookkeeping).
        primary_reward_key : str, optional
            Which objective to use as primary_reward.  Defaults to first.

        Returns
        -------
        Principle
            Populated Principle object with hypothesis and principle_text
            generated by the LLM.
        """
        primary_key = primary_reward_key or self.objective_names[0]
        primary_reward = float(outcome.get(primary_key, 0.0))

        hypothesis    = self._generate_hypothesis(sequence, outcome)
        principle_text = self._extract_principle(hypothesis, outcome)

        return Principle(
            sequence=sequence,
            outcome=dict(outcome),
            primary_reward=primary_reward,
            hypothesis=hypothesis,
            principle_text=principle_text,
            iteration=iteration,
        )

    def unload_model(self) -> None:
        """Free GPU memory occupied by the local model (e.g. before heavy GP training)."""
        self._backend.unload()

    # ------------------------------------------------------------------
    # Broad principle generation (warm-start phase 1)
    # ------------------------------------------------------------------

    def generate_broad_principles(self, n_principles: int = 5) -> List[str]:
        """
        Generate broad, domain-knowledge-based principles without any data.

        These serve as the initial seed principles that will be refined
        with training data evidence.

        Parameters
        ----------
        n_principles : int
            Number of broad principles to generate.

        Returns
        -------
        List[str]
            List of broad principle texts.
        """
        user_prompt = (
            f"Task context:\n{self.task_context}\n\n"
            f"Objectives to optimise: {', '.join(self.objective_names)}\n\n"
            f"Generate exactly {n_principles} broad scientific principles that "
            f"govern enzyme performance for this reaction. Each principle should "
            f"capture a distinct biochemical mechanism (e.g., active-site geometry, "
            f"hydrophobic core packing, electrostatic stabilisation, conformational "
            f"flexibility, substrate binding). Number them 1 through {n_principles}."
        )

        result = self._backend.generate(
            system_prompt=_BROAD_PRINCIPLE_SYSTEM,
            user_prompt=user_prompt,
            max_new_tokens=1024,
        )
        if not result:
            raise RuntimeError("PrincipleExtractor: backend returned empty broad principles.")

        # Parse numbered list
        principles = []
        for line in result.strip().split("\n"):
            line = line.strip()
            if not line:
                continue
            # Remove numbering prefix like "1." or "1)"
            for prefix_end in (". ", ") "):
                idx = line.find(prefix_end)
                if idx != -1 and idx < 5 and line[:idx].strip().isdigit():
                    line = line[idx + len(prefix_end):]
                    break
            if len(line) > 20:  # filter out very short noise lines
                principles.append(line)

        logger.info("Generated %d broad principles from domain knowledge.", len(principles))
        return principles[:n_principles]

    # ------------------------------------------------------------------
    # Evidence-based principle refinement (warm-start phase 2)
    # ------------------------------------------------------------------

    def refine_principle_with_evidence(
        self,
        principle_text: str,
        supporting_evidence: List[Dict],
        contradicting_evidence: List[Dict],
    ) -> str:
        """
        Refine a broad principle using supporting and contradicting evidence
        from the training set.

        Mirrors PiFlow's approach: present the principle alongside experimental
        evidence and ask the LLM to produce a more specific, grounded version.

        Parameters
        ----------
        principle_text : str
            The broad principle to refine.
        supporting_evidence : List[Dict]
            Training examples that support the principle. Each dict has keys:
            'sequence_desc' (str), 'outcome' (Dict[str, float]).
        contradicting_evidence : List[Dict]
            Training examples that contradict the principle.

        Returns
        -------
        str
            The refined principle text.
        """
        # Format evidence
        support_lines = []
        for i, ev in enumerate(supporting_evidence[:5], 1):  # limit to 5
            outcome_str = ", ".join(f"{k}={v:.4f}" for k, v in ev["outcome"].items())
            support_lines.append(f"  {i}. {ev['sequence_desc'][:200]}\n     Outcome: {outcome_str}")
        support_text = "\n".join(support_lines) if support_lines else "  (no strong supporting evidence found)"

        contra_lines = []
        for i, ev in enumerate(contradicting_evidence[:5], 1):
            outcome_str = ", ".join(f"{k}={v:.4f}" for k, v in ev["outcome"].items())
            contra_lines.append(f"  {i}. {ev['sequence_desc'][:200]}\n     Outcome: {outcome_str}")
        contra_text = "\n".join(contra_lines) if contra_lines else "  (no strong contradicting evidence found)"

        user_prompt = (
            f"ORIGINAL PRINCIPLE:\n{principle_text}\n\n"
            f"SUPPORTING EVIDENCE (sequences where the principle appears to hold):\n"
            f"{support_text}\n\n"
            f"CONTRADICTING EVIDENCE (sequences where the principle appears violated):\n"
            f"{contra_text}\n\n"
            f"Based on this evidence, refine the original principle to be more "
            f"specific and accurate. Incorporate quantitative patterns from the "
            f"evidence. Note boundary conditions or exceptions. Use the 3-part "
            f"structure: (1) Major premise, (2) Minor premise with evidence, "
            f"(3) Updated predictive conclusion."
        )

        result = self._backend.generate(
            system_prompt=_REFINE_PRINCIPLE_SYSTEM,
            user_prompt=user_prompt,
            max_new_tokens=512,
        )
        if not result:
            logger.warning("Refinement failed for principle: %s…; keeping original.", principle_text[:60])
            return principle_text
        return result

    # ------------------------------------------------------------------
    # Predictive hypothesis generation (inner PiFlow loop)
    # ------------------------------------------------------------------

    def generate_prediction_hypothesis(
        self,
        sequence: str,
        guidance: str,
    ) -> str:
        """
        Generate a PREDICTIVE hypothesis about a candidate sequence,
        guided by the Planner's instructions and current principles.

        This mirrors PiFlow's Hypothesis Agent: the LLM predicts how the
        sequence will perform and explains why, grounded in principles.

        Parameters
        ----------
        sequence : str
            Candidate amino acid sequence to hypothesise about.
        guidance : str
            Planner-generated guidance including action type, best principles,
            and instructions.

        Returns
        -------
        str
            The predictive hypothesis text.
        """
        seq_desc = describe_sequence(sequence)

        user_prompt = (
            f"Task context:\n{self.task_context}\n\n"
            f"PLANNER GUIDANCE:\n{guidance}\n\n"
            f"CANDIDATE SEQUENCE:\n{seq_desc}\n\n"
            f"Full sequence: {sequence}\n\n"
            f"Based on the planner's guidance and the scientific principles above, "
            f"generate a predictive hypothesis about this sequence's performance. "
            f"Explain WHY you predict it will perform well or poorly, grounding your "
            f"reasoning in the principles and specific sequence features. "
            f"Be specific about which features support or undermine performance."
        )

        result = self._backend.generate(
            system_prompt=_PREDICTION_HYPOTHESIS_SYSTEM,
            user_prompt=user_prompt,
            max_new_tokens=512,
        )
        if not result:
            raise RuntimeError(
                f"PrincipleExtractor: backend returned empty prediction hypothesis "
                f"for sequence {sequence[:40]}…"
            )
        return result

    # ------------------------------------------------------------------
    # Post-acquisition 3-agent pipeline
    # ------------------------------------------------------------------

    def rerank_principles(
        self,
        buffer: PrincipleBuffer,
        iteration_data: Dict,
    ) -> List[Principle]:
        """
        Agent 1 (Planner): Re-rank existing principles based on new iteration data.

        Parameters
        ----------
        buffer : PrincipleBuffer
            Current principle buffer.
        iteration_data : dict
            Keys:
              'gp_hypotheses' — list of {hypothesis, outcome} dicts from inner loop
              'best_so_far'   — optional dict of {objective: value}

        Returns
        -------
        List[Principle]
            Principles reordered from most to least relevant.
        """
        principles = buffer.get_all()
        if not principles:
            return []

        principle_block = "\n".join(
            f"[{i}] source={p.source} reward={p.primary_reward:.4f}\n"
            f"    {p.principle_text[:200]}"
            for i, p in enumerate(principles)
        )

        evidence_lines = []
        for ev in iteration_data.get("gp_hypotheses", [])[:5]:
            outcome_str = ", ".join(
                f"{k}={v:.4f}" for k, v in ev.get("outcome", {}).items()
            )
            evidence_lines.append(
                f"  Hypothesis: {ev.get('hypothesis', '')[:150]}\n"
                f"  GP outcome: {outcome_str}"
            )
        evidence_block = (
            "\n".join(evidence_lines) if evidence_lines
            else "  (no new GP evidence this iteration)"
        )

        best_so_far = iteration_data.get("best_so_far", {})
        gap_str = (
            ", ".join(f"{k}={v:.4f}" for k, v in best_so_far.items())
            if best_so_far else "not provided"
        )

        user_prompt = (
            f"Task context:\n{self.task_context}\n\n"
            f"CURRENT PRINCIPLES (indexed 0 to {len(principles) - 1}):\n"
            f"{principle_block}\n\n"
            f"NEW GP-VALIDATED EVIDENCE FROM THIS ITERATION:\n{evidence_block}\n\n"
            f"CURRENT BEST-SO-FAR: {gap_str}\n\n"
            f"Re-rank ALL {len(principles)} principles from most to least relevant "
            f"for guiding the next optimisation step. Follow the Required Response "
            f"Structure exactly."
        )

        result = self._backend.generate(
            system_prompt=_RERANK_PRINCIPLES_SYSTEM,
            user_prompt=user_prompt,
            max_new_tokens=512,
        )

        # Parse "Re-ranked Indices:" line — expect "3,1,0,2,4"
        ranked_indices = None
        if result:
            for line in result.split("\n"):
                if "re-ranked indices" in line.lower() or "reranked indices" in line.lower():
                    m = re.search(r"[\d,\s]+", line.split(":", 1)[-1])
                    if m:
                        try:
                            ranked_indices = [
                                int(x.strip())
                                for x in m.group().split(",")
                                if x.strip().isdigit()
                            ]
                        except ValueError:
                            pass
                    break

        if not ranked_indices:
            logger.warning(
                "rerank_principles: could not parse ranked indices from LLM response; "
                "falling back to reward order."
            )
            return sorted(principles, key=lambda p: p.primary_reward, reverse=True)

        valid = [i for i in ranked_indices if 0 <= i < len(principles)]
        missing = [i for i in range(len(principles)) if i not in valid]
        valid.extend(missing)
        logger.info("rerank_principles: new order = %s", valid)
        return [principles[i] for i in valid]

    def generate_directional_hypothesis(
        self,
        ranked_principles: List[Principle],
        experimental_data: Optional[List[Dict]] = None,
    ) -> str:
        """
        Agent 2 (Hypothesis): Generate a directional hypothesis for the next candidate.

        Parameters
        ----------
        ranked_principles : List[Principle]
            Principles ordered by current relevance (most relevant first).
        experimental_data : List[dict], optional
            Recent oracle evaluations: [{sequence_desc, outcome}].

        Returns
        -------
        str
            Structured directional hypothesis (Rationale / Hypothesis /
            Reiterate / Ideal Candidate Profile).
        """
        top = ranked_principles[:5]
        principle_block = "\n".join(
            f"{i + 1}. [reward={p.primary_reward:.4f}, source={p.source}] "
            f"{p.principle_text[:200]}"
            for i, p in enumerate(top)
        )

        exp_block = ""
        if experimental_data:
            exp_lines = []
            for ev in experimental_data[-5:]:
                outcome_str = ", ".join(
                    f"{k}={v:.4f}" for k, v in ev.get("outcome", {}).items()
                )
                exp_lines.append(
                    f"  Sequence profile: {ev.get('sequence_desc', '')[:150]}\n"
                    f"  Observed outcome: {outcome_str}"
                )
            exp_block = (
                "\n\nRECENT ORACLE OBSERVATIONS (ground-truth experimental data):\n"
                + "\n".join(exp_lines)
            )

        user_prompt = (
            f"Task context:\n{self.task_context}\n\n"
            f"TOP-RANKED PRINCIPLES (most relevant first):\n{principle_block}"
            f"{exp_block}\n\n"
            f"Based on these principles and observations, generate a directional "
            f"hypothesis for the next optimal enzyme candidate. Follow the Required "
            f"Output Format exactly."
        )

        result = self._backend.generate(
            system_prompt=_DIRECTIONAL_HYPOTHESIS_SYSTEM,
            user_prompt=user_prompt,
            max_new_tokens=512,
        )

        if not result:
            best = ranked_principles[0] if ranked_principles else None
            fallback = best.principle_text if best else "Prioritize high-reward sequence features."
            logger.warning("generate_directional_hypothesis: empty response; using fallback.")
            return fallback

        return result

    def score_candidates_by_hypothesis(
        self,
        candidates: List[str],
        hypothesis: str,
        gp_means: np.ndarray,
        gp_stds: np.ndarray,
        acq_scores: np.ndarray,
    ) -> np.ndarray:
        """
        Agent 3 (Scorer): Re-rank top-k candidates by statistical + scientific alignment.

        Parameters
        ----------
        candidates : List[str]
            Amino acid sequences of the top-k candidates.
        hypothesis : str
            Directional hypothesis from Agent 2.
        gp_means, gp_stds : np.ndarray
            GP predictions for each candidate. Shape (k,) or (k, n_obj).
        acq_scores : np.ndarray
            Acquisition / combined scores for each candidate. Shape (k,).

        Returns
        -------
        np.ndarray
            Combined scores (shape (k,)) — higher is better. Preserves relative
            order of the top-k while incorporating scientific alignment.
        """
        n = len(candidates)
        if n == 0:
            return acq_scores

        cand_lines = []
        for i, seq in enumerate(candidates):
            desc = describe_sequence(seq)
            mean_val = float(
                gp_means[i].mean() if hasattr(gp_means[i], "mean") else gp_means[i]
            )
            std_val = float(
                gp_stds[i].mean() if hasattr(gp_stds[i], "mean") else gp_stds[i]
            )
            cand_lines.append(
                f"Candidate {i}: [GP mean={mean_val:.4f}, GP std={std_val:.4f}, "
                f"acq_score={float(acq_scores[i]):.4f}]\n{desc}"
            )
        cand_block = "\n\n".join(cand_lines)

        user_prompt = (
            f"DIRECTIONAL HYPOTHESIS (target for next candidate):\n{hypothesis}\n\n"
            f"CANDIDATES ({n} total):\n{cand_block}\n\n"
            f"Rate each candidate's scientific alignment with the directional "
            f"hypothesis from 0 to 10."
        )

        result = self._backend.generate(
            system_prompt=_CANDIDATE_SCORER_SYSTEM,
            user_prompt=user_prompt,
            max_new_tokens=256,
        )

        llm_scores = np.zeros(n, dtype=np.float64)
        parsed_any = False
        if result:
            for line in result.strip().split("\n"):
                m = re.search(r"Candidate\s+(\d+)\s*:\s*([\d.]+)", line, re.IGNORECASE)
                if m:
                    idx, score = int(m.group(1)), float(m.group(2))
                    if 0 <= idx < n:
                        llm_scores[idx] = min(score, 10.0)
                        parsed_any = True

        if not parsed_any:
            logger.warning(
                "score_candidates_by_hypothesis: failed to parse LLM scores; "
                "returning original acq_scores."
            )
            return acq_scores.copy()

        def _norm(arr: "np.ndarray") -> "np.ndarray":
            lo, hi = arr.min(), arr.max()
            if hi - lo < 1e-8:
                return np.ones_like(arr) * 0.5
            return (arr - lo) / (hi - lo)

        combined = 0.5 * _norm(acq_scores) + 0.5 * _norm(llm_scores)
        return combined

    # ------------------------------------------------------------------
    # Private prompt helpers
    # ------------------------------------------------------------------

    def _outcome_str(self, outcome: Dict[str, float]) -> str:
        return ", ".join(f"{k} = {v:.4f}" for k, v in outcome.items())

    def _generate_hypothesis(self, sequence: str, outcome: Dict[str, float]) -> str:
        """
        Ask the LLM: why did this sequence achieve these outcomes?
        Mirrors PiFlow's HypothesisAgent role.
        """
        seq_desc    = describe_sequence(sequence)
        outcome_str = self._outcome_str(outcome)

        user_prompt = (
            f"Task context:\n{self.task_context}\n\n"
            f"Enzyme sequence properties:\n{seq_desc}\n\n"
            f"Full sequence: "
            f"{sequence}\n\n"
            f"Observed experimental outcomes: {outcome_str}\n\n"
            f"Formulate a mechanistic hypothesis (3–5 sentences) explaining why "
            f"this enzyme achieved these results. Focus on specific sequence features "
            f"(residue composition, charge distribution, known catalytic motifs) and "
            f"their mechanistic connection to the observed yield and enantioselectivity."
        )

        result = self._backend.generate(
            system_prompt=_HYPOTHESIS_SYSTEM,
            user_prompt=user_prompt,
            max_new_tokens=512,
        )
        if not result:
            raise RuntimeError(
                f"PrincipleExtractor: backend returned empty hypothesis for sequence "
                f"{sequence[:40]}…"
            )
        return result

    def _extract_principle(
        self, hypothesis: str, outcome: Dict[str, float]
    ) -> str:
        """
        Ask the LLM: what generalizable principle does this hypothesis + result reveal?
        Mirrors PiFlow's llm_assign_principle.
        """
        outcome_str = self._outcome_str(outcome)

        user_prompt = (
            f"Hypothesis:\n{hypothesis}\n\n"
            f"Experimental outcome: {outcome_str}\n\n"
            f"Based on the hypothesis and this experimental feedback, extract a "
            f"generalizable scientific principle using the required 3-part structure:\n"
            f"(1) Major premise: the general biochemical design rule.\n"
            f"(2) Minor premise: how this sequence exemplifies it.\n"
            f"(3) Conclusion: a predictive statement for future sequence design."
        )

        result = self._backend.generate(
            system_prompt=_PRINCIPLE_SYSTEM,
            user_prompt=user_prompt,
            max_new_tokens=256,
        )
        if not result:
            raise RuntimeError(
                "PrincipleExtractor: backend returned empty principle."
            )
        return result
