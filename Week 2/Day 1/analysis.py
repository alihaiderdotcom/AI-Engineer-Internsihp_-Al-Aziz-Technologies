"""Clean and summarize a small sales dataset."""

import json
from pathlib import Path

import numpy as np
import pandas as pd


BASE_DIR = Path(__file__).parent


def clean_sales_data(path):
    sales = pd.read_csv(path)
    sales["product"] = sales["product"].fillna("Unknown").str.strip().str.title()
    sales["quantity"] = pd.to_numeric(sales["quantity"], errors="coerce").fillna(0)
    sales["price"] = pd.to_numeric(sales["price"], errors="coerce").fillna(0.0)
    sales["quantity"] = sales["quantity"].astype(int)
    sales = sales.drop_duplicates().copy()
    sales["revenue"] = sales["quantity"] * sales["price"]
    return sales.sort_values("revenue", ascending=False).reset_index(drop=True)


def summarize(sales):
    quantities = np.array(sales["quantity"], dtype=int)
    revenues = np.array(sales["revenue"], dtype=float)
    return {
        "rows_after_cleaning": int(len(sales)),
        "total_quantity": int(quantities.sum()),
        "average_quantity": round(float(quantities.mean()), 2),
        "total_revenue": round(float(revenues.sum()), 2),
        "top_product": str(sales.iloc[0]["product"]),
    }


def main():
    cleaned = clean_sales_data(BASE_DIR / "sales.csv")
    cleaned.to_csv(BASE_DIR / "cleaned_sales.csv", index=False)
    summary = summarize(cleaned)
    (BASE_DIR / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(cleaned.to_string(index=False))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
