"""Process score data from CSV and produce a JSON summary."""

import csv
import json
import logging
import os
from pathlib import Path

import numpy as np


BASE_DIR = Path(__file__).parent
logging.basicConfig(
    filename=BASE_DIR / "pipeline.log",
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
)
logger = logging.getLogger(__name__)


def load_config(path):
    """Load application settings from a JSON file."""
    try:
        with path.open("r", encoding="utf-8") as file:
            return json.load(file)
    except (OSError, json.JSONDecodeError) as error:
        logger.error("Could not load config: %s", error)
        raise


def load_scores(path):
    """Load valid score rows and skip malformed rows with a warning."""
    scores = []
    with path.open("r", newline="", encoding="utf-8") as file:
        for row_number, row in enumerate(csv.DictReader(file), start=2):
            try:
                score = float(row["score"])
                if not 0 <= score <= 100:
                    raise ValueError("score must be between 0 and 100")
                scores.append({"name": row["name"], "score": score})
            except (KeyError, TypeError, ValueError) as error:
                logger.warning("Skipping row %s: %s", row_number, error)
    return scores


def build_summary(records, app_name, passing_score):
    """Calculate summary statistics using vectorized NumPy operations."""
    if not records:
        raise ValueError("No valid score records were found")

    values = np.array([record["score"] for record in records], dtype=float)
    passed = values >= passing_score
    return {
        "application": app_name,
        "passing_score": passing_score,
        "students_processed": len(records),
        "average_score": round(float(values.mean()), 2),
        "highest_score": float(values.max()),
        "lowest_score": float(values.min()),
        "passed": int(passed.sum()),
        "failed": int((~passed).sum()),
        "scores_plus_five": (values + 5).tolist(),
    }


def save_json(path, data):
    """Write structured output as readable JSON."""
    with path.open("w", encoding="utf-8") as file:
        json.dump(data, file, indent=2)


def main():
    app_name = os.getenv("APP_NAME", "Day 4 Data Pipeline")
    config = load_config(BASE_DIR / "config.json")
    records = load_scores(BASE_DIR / "scores.csv")
    summary = build_summary(records, app_name, config["passing_score"])
    save_json(BASE_DIR / "summary.json", summary)
    logger.info("Processed %s valid records", len(records))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
