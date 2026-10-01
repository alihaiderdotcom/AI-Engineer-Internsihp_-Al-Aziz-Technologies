# Week 5 Day 2: Vector Database Fundamentals

## Objective

Persist documents, vectors, and metadata so retrieval survives beyond one process.

## Implementation

`vector_store.py` uses SQLite as a small local vector database. Each record stores source text, its embedding, and JSON metadata. Search computes cosine similarity over stored vectors and returns ranked records.

## Production Notes

SQLite is ideal for a transparent prototype and test fixture. Larger workloads should move similarity search to a purpose-built index such as pgvector, Qdrant, Milvus, or a managed vector service.

## Run

```bash
python vector_store.py
```

## Deliverable Checklist

- [x] Persistent document schema
- [x] Metadata support
- [x] Top-k similarity search
- [x] Explicit upgrade path to a production vector database
