# Week 6 Day 4: Packaging & Deployment

## Container Build

Run from the repository root:

```bash
docker build -f "Week 6/Day 4/Dockerfile" -t knowledgedesk:local .
docker run --rm -p 8080:8080 knowledgedesk:local
```

## Smoke Checks

```bash
curl http://127.0.0.1:8080/health
curl -X POST http://127.0.0.1:8080/query \\
  -H 'Content-Type: application/json' \\
  -d '{"question":"How does KnowledgeDesk cite retrieved documents?"}'
```

## Production Checklist

- Store provider credentials in a secret manager, never in the image.
- Put TLS and authentication at the edge or API gateway.
- Configure CPU and memory limits plus a restart policy.
- Send structured audit events to a central log sink.
- Monitor latency, error rate, grounded-answer rate, and citation coverage.
- Run the Week 6 evaluation before promoting a new image.

The included server is a learning implementation. A deployed service should use a production WSGI/ASGI server, authentication middleware, and a managed vector database.
