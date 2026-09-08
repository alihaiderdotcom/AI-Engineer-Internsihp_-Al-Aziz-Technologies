# Week 2, Day 2 - Data Analysis & Visualization

**Focus:** Exploratory Data Analysis, descriptive statistics, correlation, outliers, and Matplotlib.

## Learning Objectives

- Describe a dataset with mean, median, mode, variance, and standard deviation.
- Inspect relationships with correlation.
- Identify possible outliers using the IQR rule.
- Create histograms, bar charts, scatter plots, and line charts.
- Save charts as files for documentation.

## Run

```bash
.venv/bin/python eda.py
```

The script reads `sales.csv`, prints an EDA report, and writes charts to `charts/`.

## Key Concepts

- **Mean:** arithmetic average.
- **Median:** middle value after sorting.
- **Mode:** most frequent value.
- **Variance and standard deviation:** measures of spread.
- **Correlation:** strength and direction of a linear relationship.
- **IQR outlier rule:** values below $Q1 - 1.5(IQR)$ or above $Q3 + 1.5(IQR)$ are flagged for review.

## Learning Summary

Today I learned that EDA is the process of understanding data before modeling it. Statistics describe the shape of the data, while visualizations make trends, relationships, and unusual values easier to see.

## Completion Checklist

- [x] Descriptive statistics
- [x] Correlation analysis
- [x] Outlier detection
- [x] Histogram
- [x] Bar chart
- [x] Scatter plot
- [x] Line chart
