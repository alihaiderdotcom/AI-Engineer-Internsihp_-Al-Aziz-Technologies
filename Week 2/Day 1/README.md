# Week 2, Day 1 - NumPy & Pandas

**Focus:** Arrays, DataFrames, data loading, selection, filtering, sorting, and cleaning.

## Learning Objectives

- Create NumPy arrays and inspect their shape and data type.
- Use indexing, broadcasting, and vectorized arithmetic.
- Load CSV data into a Pandas DataFrame.
- Detect missing values and duplicate rows.
- Clean numeric and categorical columns.
- Filter and sort records.
- Export cleaned data and a JSON summary.

## Run

```bash
python3 analysis.py
```

The script reads `sales.csv`, cleans it, writes `cleaned_sales.csv`, and prints a summary.

## Key Examples

```python
import numpy as np
import pandas as pd

values = np.array([10, 20, 30])
print(values.shape)
print(values * 2)

sales = pd.read_csv("sales.csv")
sales = sales.drop_duplicates()
sales["quantity"] = sales["quantity"].fillna(0).astype(int)
filtered = sales[sales["quantity"] > 0]
```

## Learning Summary

Today I learned that NumPy is useful for fast numerical operations, while Pandas provides labeled, table-shaped data structures. Cleaning must happen before analysis because missing, duplicated, or incorrectly typed values can produce misleading results.

## Completion Checklist

- [x] NumPy arrays, shapes, indexing, and broadcasting
- [x] Pandas Series and DataFrames
- [x] CSV loading and export
- [x] Missing values and duplicates
- [x] Filtering and sorting
- [x] Basic data cleaning
