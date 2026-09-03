# Week 1, Day 4 - Python for AI Development

**Date:** September 4, 2026  
**Focus:** Files, CSV, JSON, exceptions, logging, environment variables, API requests, and NumPy

## Learning Objectives

By the end of today, I can:

- Read and write text, CSV, and JSON files.
- Validate input and handle expected exceptions.
- Configure useful application logging.
- Read configuration from environment variables.
- Make an HTTP request safely.
- Use NumPy arrays and vectorized operations for numerical data.

## Core Concepts

### File Handling

Use `with open(...)` so files are closed automatically:

```python
with open("notes.txt", "r", encoding="utf-8") as file:
    contents = file.read()
```

`csv.DictReader` and `csv.DictWriter` are useful when each row has named fields. The `json` module handles structured data that maps naturally to Python dictionaries and lists.

### Exceptions and Logging

```python
try:
    value = float(raw_value)
except ValueError:
    logger.warning("Invalid numeric value: %s", raw_value)
```

Exceptions should handle expected failures at the boundary where they can be explained or recovered from. Logging records what happened without relying on scattered `print` statements.

### Environment Variables

Configuration and secrets should not be hard-coded:

```bash
export APP_NAME="Day 4 Data Pipeline"
```

The exercise reads `APP_NAME` with `os.getenv` and falls back to a safe default. A real API key must never be committed to GitHub.

### HTTP Requests

The optional `api_demo.py` uses `requests.get` with a timeout and `raise_for_status()`:

```python
response = requests.get(url, timeout=10)
response.raise_for_status()
```

Always set a timeout, check the response, and handle network errors.

### NumPy

NumPy stores numerical data in efficient arrays and supports vectorized operations:

```python
scores = np.array([85, 90, 78])
adjusted = scores + 5
normalized = (scores - scores.mean()) / scores.std()
```

Vectorized operations apply to the full array without writing a manual loop.

## Practical Exercise

Run the offline pipeline:

```bash
python3 main.py
```

It reads `scores.csv`, loads settings from `config.json`, validates rows, calculates statistics with NumPy, writes `summary.json`, and logs invalid rows to `pipeline.log`.

Run the optional API exercise when network access is available:

```bash
python3 api_demo.py
```

## Files

- `main.py` - end-to-end CSV/JSON/NumPy pipeline
- `api_demo.py` - timeout-aware public API request
- `scores.csv` - sample input data
- `config.json` - non-secret application settings
- `.env.example` - example environment configuration
- `requirements.txt` - external dependencies

## Learning Summary

Today I learned how Python applications move from simple scripts toward AI-ready data pipelines. I practiced loading structured data, validating it, recording failures with logging, keeping configuration outside the code, calling an API safely, and using NumPy for fast numerical calculations.

## Completion Checklist

- [x] File handling
- [x] CSV and JSON
- [x] Exception handling
- [x] Logging basics
- [x] Environment variables
- [x] HTTP requests
- [x] NumPy arrays and vectorized operations
- [x] Working data-processing pipeline
