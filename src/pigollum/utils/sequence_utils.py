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
