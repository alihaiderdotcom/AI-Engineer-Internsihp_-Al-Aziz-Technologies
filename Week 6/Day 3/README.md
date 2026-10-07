# Week 6 Day 3: Reliability, Security & Observability

`production.py` demonstrates reusable service controls:

- redact likely secrets before logging,
- limit requests per client in a time window,
- generate request identifiers,
- hash request content instead of storing raw prompts,
- emit compact audit events with grounding and citation fields.

```bash
python production.py
```

These primitives complement, rather than replace, provider authentication, TLS, centralized logging, and a managed rate limiter.
