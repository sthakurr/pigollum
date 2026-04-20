#!/usr/bin/env python3
"""Quick script to query Qwen/Qwen2.5-7B-Instruct via HuggingFace transformers."""
import sys
from transformers import AutoModelForCausalLM, AutoTokenizer

"""
Utilities for working with amino acid sequences in PiGollum.

These helpers convert raw sequences into human-readable descriptions
that are suitable for LLM-based hypothesis and principle generation.
"""
from typing import Dict

# Standard amino acid properties
_HYDROPHOBIC = set("AILMFWVP")
_CHARGED_POS = set("RKH")
_CHARGED_NEG = set("DE")
_POLAR       = set("STNQ")
_SPECIAL     = set("CGY")  # cysteine, glycine, tyrosine

_PROPERTY_MAP: Dict[str, str] = {
    "A": "alanine (small, nonpolar)",
    "R": "arginine (positively charged)",
    "N": "asparagine (polar)",
    "D": "aspartate (negatively charged)",
    "C": "cysteine (special, can form disulfide bonds)",
    "Q": "glutamine (polar)",
    "E": "glutamate (negatively charged)",
    "G": "glycine (special, flexible)",
    "H": "histidine (aromatic, can be positively charged)",
    "I": "isoleucine (hydrophobic)",
    "L": "leucine (hydrophobic)",
    "K": "lysine (positively charged)",
    "M": "methionine (hydrophobic, initiator)",
    "F": "phenylalanine (aromatic, hydrophobic)",
    "P": "proline (special, helix-breaker)",
    "S": "serine (polar, phosphorylatable)",
    "T": "threonine (polar, phosphorylatable)",
    "W": "tryptophan (aromatic, hydrophobic)",
    "Y": "tyrosine (aromatic, polar)",
    "V": "valine (hydrophobic)",
}


def amino_acid_composition(sequence: str) -> Dict[str, float]:
    """Return fractional composition of amino acid property groups."""
    seq = sequence.upper()
    n = len(seq)
    if n == 0:
        return {}
    return {
        "hydrophobic_frac":    sum(1 for aa in seq if aa in _HYDROPHOBIC)    / n,
        "pos_charged_frac":    sum(1 for aa in seq if aa in _CHARGED_POS)    / n,
        "neg_charged_frac":    sum(1 for aa in seq if aa in _CHARGED_NEG)    / n,
        "polar_frac":          sum(1 for aa in seq if aa in _POLAR)          / n,
        "special_frac":        sum(1 for aa in seq if aa in _SPECIAL)        / n,
        "length":              float(n),
        "net_charge_estimate": (
            sum(1 for aa in seq if aa in _CHARGED_POS) -
            sum(1 for aa in seq if aa in _CHARGED_NEG)
        ),
    }


def describe_sequence(sequence: str, max_show: int = 60) -> str:
    """
    Produce a short human-readable description of an amino acid sequence
    suitable for inclusion in LLM prompts.

    Parameters
    ----------
    sequence : str
        Full amino acid sequence (single-letter codes).
    max_show : int
        Number of leading residues to show verbatim.

    Returns
    -------
    str
        Multi-line description string.
    """
    seq = sequence.upper()
    comp = amino_acid_composition(seq)
    n = int(comp.get("length", len(seq)))

    truncated = seq[:max_show] + ("…" if n > max_show else "")

    lines = [
        f"Length: {n} amino acids",
        f"Sequence (first {min(n, max_show)} residues): {truncated}",
        f"Composition: "
        f"{comp['hydrophobic_frac']*100:.1f}% hydrophobic, "
        f"{comp['pos_charged_frac']*100:.1f}% positive-charged, "
        f"{comp['neg_charged_frac']*100:.1f}% negative-charged, "
        f"{comp['polar_frac']*100:.1f}% polar",
        f"Estimated net charge: {comp['net_charge_estimate']:+.0f}",
    ]
    return "\n".join(lines)


MODEL = "Qwen/Qwen2.5-7B-Instruct"

tokenizer = AutoTokenizer.from_pretrained(MODEL)
model = AutoModelForCausalLM.from_pretrained(MODEL, torch_dtype="auto", device_map="auto")

task_context = "Biocatalytic enzyme engineering: optimise amino acid sequences (enzymes) to maximise reaction yield and enantioselectivity (ddg_scaled) for a target asymmetric transformation. The sequences are not really related to each other as we are in an exploration phase but we need to train an intelligent GP that can learn the defining principles for this task from the weak signal we have. We need the GP to have enough information to rank a subset of sequences to perform virtual screening. The top-ranked sequences will go further for experimental validation."
reaction_smiles = "CC(C(=O)ON1C(=O)C2=C(C=CC=C2)C1=O)C1=CC=CC=C1 + TMSCN --> CC(C#N)C1=CC=CC=C1"
sequence = "TPEEEEVVAAIREIAPEADVEAAFDAAAAAMGVTREELFAMLDQPVGSLPPERVDAFIDGMAEMSVLLVGADPEVGARARARAAERVRAEVRPEDHMWRVMAMVHRLVAEALREMGHPEAAKALAIAEVVERAAARMLAAL"
outcome_str = "yield=59%, no enantioselectivity (ddg_scaled=0.0)"  # Example observed outcome for this sequence
seq_desc = describe_sequence(sequence)
user_prompt = f"Task context:\n{task_context}\n\nTarget reaction: {reaction_smiles}\n\nFull sequence: {sequence}\n\nObserved experimental outcomes: {outcome_str}\n\nFormulate a single-sentence mechanistic hypothesis explaining why this enzyme achieved these results. The hypothesis must name a specific sequence feature (e.g. a particular residue type, motif, or charge pattern) and link it directly to the observed yield and enantioselectivity. Output only the hypothesis sentence, nothing else."
# prompt = " ".join(sys.argv[1:]) if len(sys.argv) > 1 else input("Prompt: ")

messages = [{"role": "user", "content": user_prompt}]
text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
inputs = tokenizer([text], return_tensors="pt").to(model.device)

output_ids = model.generate(**inputs, max_new_tokens=512)
# Trim the input tokens from the output
generated = output_ids[0][inputs.input_ids.shape[1]:]
print(tokenizer.decode(generated, skip_special_tokens=True))
