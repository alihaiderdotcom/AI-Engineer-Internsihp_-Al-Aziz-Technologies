# Week 2, Day 5 - Weekly Project & Revision

**Focus:** End-to-end machine learning pipeline combining data cleaning, EDA, model training, and evaluation.

## Weekly Project: Iris Flower Classification

This comprehensive project demonstrates a complete ML workflow from raw data to model evaluation. It integrates:

- Data loading and cleaning with Pandas (Day 1)
- Exploratory data analysis and visualization (Day 2)
- Classification and preprocessing (Day 3)
- Model evaluation and comparison (Day 4)

## Run

```bash
.venv/bin/python iris_project.py
```

The script:

1. Loads the Iris dataset
2. Performs data exploration
3. Trains multiple classifiers
4. Compares performance
5. Generates a project report

## Project Components

### Phase 1: Data & EDA

- Load and inspect the dataset
- Check for missing values and duplicates
- Print descriptive statistics
- Visualize feature distributions and relationships

### Phase 2: Preprocessing

- Separate features from labels
- Standardize features
- Split into training and test sets

### Phase 3: Model Training

- Train Logistic Regression, Decision Tree, and Random Forest
- Compare metrics across models
- Display confusion matrices

### Phase 4: Report

- Print summary of best model
- Save results to `iris_results.txt`

## Expected Output

The report includes dataset shape, feature statistics, model accuracies, precision/recall, and the best-performing model recommendation.

## Learning Summary (In Your Own Words)

This week taught me that machine learning is a complete pipeline, not just model training. Data quality affects everything downstream. EDA reveals patterns and issues. Preprocessing must respect train/test boundaries. Model comparison requires multiple metrics. Documentation helps explain decisions and findings to others.

## Revision Checklist

- [x] Data loading and exploration with Pandas and NumPy
- [x] Descriptive statistics and correlation
- [x] Data visualization
- [x] Data preprocessing and scaling
- [x] Train/test split
- [x] Multiple model training
- [x] Model evaluation metrics
- [x] Confusion matrices
- [x] Report generation

## Week 2 Deliverables

- Working ML project with documentation
- Code is clean, commented, and modular
- README explaining the workflow
- Generated results report
- GitHub-ready folder structure
