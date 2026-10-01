# Week 5 Day 1: Embeddings & Semantic Search

## Objective

Understand how text becomes a numeric representation and how vector similarity supports semantic retrieval.

## Implementation

`embeddings.py` provides a deterministic hashed bag-of-words embedding, cosine similarity, and ranked search. It is intentionally transparent and offline; production systems should use a trained embedding model.

## Concepts

- Tokenization and normalization
- Vector dimensions and normalization
- Cosine similarity
- Approximate nearest-neighbor trade-offs
- Why embedding quality matters more than the storage layer

## Run

```bash
python embeddings.py
```

## Deliverable Checklist

- [x] Deterministic embedding function
- [x] Cosine similarity implementation
- [x] Ranked semantic search demo
- [x] Limitation documented: hashed vectors are a teaching baseline
