"""Offline embedding and semantic-search fundamentals."""

from __future__ import annotations

import hashlib
import math
import re
from dataclasses import dataclass
from typing import Iterable, List, Sequence, Tuple

TOKEN_RE = re.compile(r"[a-z0-9]+")


def tokenize(text: str) -> List[str]:
    return TOKEN_RE.findall(text.lower())


def embed(text: str, dimensions: int = 64) -> List[float]:
    """Create a deterministic hashed bag-of-words vector.

    This is a teaching baseline, not a replacement for a trained embedding model.
    """
    vector = [0.0] * dimensions
    tokens = tokenize(text)
    for token in tokens:
        digest = hashlib.sha256(token.encode("utf-8")).digest()
        index = int.from_bytes(digest[:4], "big") % dimensions
        vector[index] += 1.0
    norm = math.sqrt(sum(value * value for value in vector))
    return [value / norm for value in vector] if norm else vector


def cosine_similarity(left: Sequence[float], right: Sequence[float]) -> float:
    if len(left) != len(right):
        raise ValueError("Vectors must have the same dimensions")
    return sum(a * b for a, b in zip(left, right))


@dataclass(frozen=True)
class SearchResult:
    text: str
    score: float


def semantic_search(query: str, documents: Iterable[str], top_k: int = 3) -> List[SearchResult]:
    query_vector = embed(query)
    ranked = [SearchResult(text, cosine_similarity(query_vector, embed(text))) for text in documents]
    return sorted(ranked, key=lambda result: result.score, reverse=True)[:top_k]


def main() -> None:
    documents = [
        "VisionFlow uses a PyTorch CNN for image classification.",
        "KnowledgeDesk cites retrieved documents in every grounded answer.",
        "SQLite provides a lightweight persistent store for local prototypes.",
    ]
    print("Query: How does KnowledgeDesk provide grounded answers?")
    for result in semantic_search("How does KnowledgeDesk provide grounded answers?", documents):
        print(f"{result.score:.3f}  {result.text}")


if __name__ == "__main__":
    main()
