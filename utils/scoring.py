from __future__ import annotations


def compute_round_score(correct: bool) -> dict:
    """Return score breakdown for a round."""
    base = 100 if correct else 0
    bonus = 0
    penalty = 0
    total = base

    return {
        "base": base,
        "bonus": bonus,
        "penalty": penalty,
        "points": total,
    }
