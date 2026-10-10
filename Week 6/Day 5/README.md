# Week 6 Day 5: KnowledgeDesk Master Project

## Final Project

The master project packages the Week 5 assistant as a production-shaped service and runs its regression suite. `master_project.py` generates `master_project_report.json` with capabilities, timestamp, and evaluation results.

## Run

```bash
python master_project.py
```

A successful run requires all three golden cases to pass: cited retrieval, calculator tool use, and abstention for unsupported evidence.

## Completion Checklist

- [x] Reusable retrieval and tool components
- [x] HTTP health and query endpoints
- [x] Security and observability primitives
- [x] Container deployment artifact
- [x] Automated evaluation report
- [x] Reproducible master project runner
