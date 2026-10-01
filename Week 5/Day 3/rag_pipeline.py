"""Cited, offline Retrieval-Augmented Generation pipeline."""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "Day 1"))
from embeddings import cosine_similarity, embed

STOPWORDS = {"a", "an", "and", "are", "does", "how", "is", "of", "the", "what", "where"}


@dataclass(frozen=True)
class Document:
    document_id: str
    title: str
    text: str


@dataclass(frozen=True)
class RetrievedDocument:
    document: Document
    score: float


class RAGPipeline:
    def __init__(self, documents: Iterable[Document], minimum_score: float = 0.15) -> None:
        self.documents = list(documents)
        self.minimum_score = minimum_score

    def retrieve(self, query: str, top_k: int = 3) -> List[RetrievedDocument]:
        query_vector = embed(query)
        query_terms = set(re.findall(r"[a-z0-9]+", query.lower())) - STOPWORDS
        ranked = []
        for document in self.documents:
            document_terms = set(re.findall(r"[a-z0-9]+", document.text.lower()))
            if not query_terms.intersection(document_terms):
                continue
            ranked.append(RetrievedDocument(document, cosine_similarity(query_vector, embed(document.text))))
        return [item for item in sorted(ranked, key=lambda item: item.score, reverse=True)[:top_k] if item.score >= self.minimum_score]

    def answer(self, query: str, top_k: int = 3) -> dict:
        retrieved = self.retrieve(query, top_k)
        if not retrieved:
            return {"answer": "I do not have enough evidence in the indexed documents.", "citations": [], "grounded": False}
        evidence = " ".join(item.document.text for item in retrieved)
        query_terms = set(re.findall(r"[a-z0-9]+", query.lower())) - STOPWORDS
        sentences = re.split(r"(?<=[.!?])\s+", evidence)
        selected = [sentence for sentence in sentences if query_terms.intersection(sentence.lower().split())]
        answer = " ".join(selected[:2]) or evidence
        return {
            "answer": answer,
            "citations": [item.document.document_id for item in retrieved],
            "grounded": True,
            "scores": {item.document.document_id: round(item.score, 3) for item in retrieved},
        }


def default_documents() -> List[Document]:
    return [
        Document("week-3-visionflow", "VisionFlow", "VisionFlow is a PyTorch CNN project that achieved 94.2% test accuracy."),
        Document("week-5-rag", "RAG", "KnowledgeDesk retrieves relevant documents and includes their identifiers as citations."),
        Document("week-5-storage", "Storage", "The local vector store uses SQLite to persist document text, embeddings, and metadata."),
    ]


def main() -> None:
    pipeline = RAGPipeline(default_documents())
    print(pipeline.answer("How does KnowledgeDesk cite retrieved documents?"))
    print(pipeline.answer("What is the moon made of?"))


if __name__ == "__main__":
    main()
