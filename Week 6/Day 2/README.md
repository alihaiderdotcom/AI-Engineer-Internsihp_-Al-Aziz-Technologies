# Week 6 Day 2: HTTP API Engineering

`service.py` exposes a dependency-free JSON service:

- `GET /health` returns service status.
- `POST /query` accepts `{ "question": "..." }` and returns a KnowledgeDesk response.

Run locally with:

```bash
python service.py --host 127.0.0.1 --port 8080
python service.py --demo
```

The standard-library server is intentionally easy to inspect. FastAPI, authentication, and an ASGI server are documented as production upgrades.
