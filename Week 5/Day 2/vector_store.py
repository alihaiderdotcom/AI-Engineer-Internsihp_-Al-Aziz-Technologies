"""A small persistent vector store built on SQLite and hashed embeddings."""

from __future__ import annotations

import json
import math
import sqlite3
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Sequence

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "Day 1"))
from embeddings import embed


@dataclass(frozen=True)
class VectorRecord:
    record_id: int
    text: str
    metadata: dict
    score: float


class SQLiteVectorStore:
    def __init__(self, database_path: str = ":memory:") -> None:
        self.connection = sqlite3.connect(database_path)
        self.connection.execute(
            "CREATE TABLE IF NOT EXISTS documents (id INTEGER PRIMARY KEY, text TEXT NOT NULL, embedding TEXT NOT NULL, metadata TEXT NOT NULL)"
        )
        self.connection.commit()

    def add(self, text: str, metadata: dict | None = None) -> int:
        cursor = self.connection.execute(
            "INSERT INTO documents(text, embedding, metadata) VALUES (?, ?, ?)",
            (text, json.dumps(embed(text)), json.dumps(metadata or {})),
        )
        self.connection.commit()
        return int(cursor.lastrowid)

    def search(self, query: str, top_k: int = 3) -> List[VectorRecord]:
        query_vector = embed(query)
        results: List[VectorRecord] = []
        for record_id, text, embedding_json, metadata_json in self.connection.execute(
            "SELECT id, text, embedding, metadata FROM documents"
        ):
            vector = json.loads(embedding_json)
            score = sum(a * b for a, b in zip(query_vector, vector))
            results.append(VectorRecord(record_id, text, json.loads(metadata_json), score))
        return sorted(results, key=lambda result: result.score, reverse=True)[:top_k]

    def close(self) -> None:
        self.connection.close()


def main() -> None:
    store = SQLiteVectorStore()
    store.add("KnowledgeDesk answers questions from indexed internship documents.", {"source": "knowledge-desk"})
    store.add("The vector store persists text, embeddings, and metadata in SQLite.", {"source": "storage"})
    store.add("VisionFlow is a PyTorch computer vision project.", {"source": "week-3"})
    for result in store.search("Where are embeddings stored?"):
        print(f"{result.score:.3f}  [{result.metadata['source']}] {result.text}")
    store.close()


if __name__ == "__main__":
    main()
