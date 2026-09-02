"""Reusable calculation functions for the Day 3 exercise."""


def calculate_average(scores, precision=2):
    """Return the rounded average for a non-empty collection of scores."""
    if not scores:
        raise ValueError("scores must contain at least one value")
    return round(sum(scores) / len(scores), precision)


def get_grade(score):
    """Convert an average score into a letter grade."""
    if not 0 <= score <= 100:
        raise ValueError("score must be between 0 and 100")

    if score >= 90:
        return "A"
    if score >= 80:
        return "B"
    if score >= 70:
        return "C"
    if score >= 60:
        return "D"
    return "F"


def summarize_scores(*scores, **metadata):
    """Return a summary using flexible positional and named arguments."""
    return {
        "average": calculate_average(scores),
        "count": len(scores),
        "metadata": metadata,
    }
