# Week 5: RAG, Vector Databases, Agents & APIs

Week 5 turns the Week 4 LLM foundations into grounded applications. The exercises build one small production-shaped system, **KnowledgeDesk**, from deterministic local retrieval and citations through tool use and an HTTP API.

## Learning Path

| Day | Focus | Deliverable |
|---|---|---|
| 1 | Embeddings and semantic search | `Day 1/embeddings.py` |
| 2 | Vector database fundamentals | `Day 2/vector_store.py` |
| 3 | Retrieval-Augmented Generation | `Day 3/rag_pipeline.py` |
| 4 | Tool-using AI agents | `Day 4/agent.py` |
| 5 | Weekly integration project | `Day 5/knowledge_desk.py` |

## Design Principles

- Retrieval is deterministic and inspectable before an LLM is introduced.
- Every generated answer carries document citations.
- Tools are explicitly registered, validated, and executed outside the model.
- The examples run offline with the Python standard library.
- External providers can be added behind the same interfaces later.

## Run

```bash
python "Week 5/Day 1/embeddings.py"
python "Week 5/Day 2/vector_store.py"
python "Week 5/Day 3/rag_pipeline.py"
python "Week 5/Day 4/agent.py"
python "Week 5/Day 5/knowledge_desk.py"
```

## Weekly Deliverables

- [x] Embedding and cosine-similarity baseline
- [x] Persistent SQLite vector store
- [x] Cited RAG pipeline with abstention when evidence is missing
- [x] Allow-listed agent tools and bounded execution
- [x] Integrated KnowledgeDesk demonstration and documentation
