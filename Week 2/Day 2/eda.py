"""Perform exploratory data analysis and save Matplotlib charts."""

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


BASE_DIR = Path(__file__).parent
CHART_DIR = BASE_DIR / "charts"


def find_outliers(series):
    first_quartile = series.quantile(0.25)
    third_quartile = series.quantile(0.75)
    iqr = third_quartile - first_quartile
    lower_bound = first_quartile - 1.5 * iqr
    upper_bound = third_quartile + 1.5 * iqr
    return series[(series < lower_bound) | (series > upper_bound)]


def print_report(sales):
    print("Dataset shape:", sales.shape)
    print("\nDescriptive statistics:")
    print(sales[["quantity", "price", "revenue"]].describe().round(2))
    print("\nMedian values:")
    print(sales[["quantity", "price", "revenue"]].median().round(2))
    print("\nMost common products:")
    print(sales["product"].mode().tolist())
    print("\nCorrelation matrix:")
    print(sales[["quantity", "price", "revenue"]].corr().round(2))
    print("\nRevenue outliers:")
    print(find_outliers(sales["revenue"]).to_string())


def save_charts(sales):
    CHART_DIR.mkdir(exist_ok=True)

    plt.figure(figsize=(8, 5))
    sales["revenue"].plot.hist(bins=5, color="#2f6690", edgecolor="white")
    plt.title("Revenue Distribution")
    plt.xlabel("Revenue")
    plt.tight_layout()
    plt.savefig(CHART_DIR / "revenue_histogram.png")
    plt.close()

    product_revenue = sales.groupby("product")["revenue"].sum().sort_values()
    plt.figure(figsize=(8, 5))
    product_revenue.plot.barh(color="#3d9970")
    plt.title("Revenue by Product")
    plt.xlabel("Revenue")
    plt.tight_layout()
    plt.savefig(CHART_DIR / "product_bar_chart.png")
    plt.close()

    plt.figure(figsize=(8, 5))
    plt.scatter(sales["quantity"], sales["revenue"], color="#c44e52")
    plt.title("Quantity and Revenue")
    plt.xlabel("Quantity")
    plt.ylabel("Revenue")
    plt.tight_layout()
    plt.savefig(CHART_DIR / "quantity_revenue_scatter.png")
    plt.close()

    daily_revenue = sales.groupby("day")["revenue"].sum().sort_index()
    plt.figure(figsize=(8, 5))
    daily_revenue.plot(marker="o", color="#8172b2")
    plt.title("Daily Revenue")
    plt.xlabel("Day")
    plt.ylabel("Revenue")
    plt.tight_layout()
    plt.savefig(CHART_DIR / "daily_revenue_line.png")
    plt.close()


def main():
    sales = pd.read_csv(BASE_DIR / "sales.csv")
    print_report(sales)
    save_charts(sales)
    print(f"\nSaved charts to {CHART_DIR}")


if __name__ == "__main__":
    main()
