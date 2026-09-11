# Week 2, Day 4 - Scikit-learn & Model Evaluation

**Focus:** Train multiple models, compare metrics, inspect confusion matrices, and understand overfitting and cross-validation.

## Learning Objectives

- Train and compare Logistic Regression, Decision Tree, Random Forest, and K-Nearest Neighbors models.
- Measure accuracy, precision, recall, and F1-score.
- Read a confusion matrix.
- Calculate mean squared error for a regression model.
- Compare training and test accuracy as a simple overfitting signal.
- Use stratified cross-validation for a more reliable estimate.

## Run

```bash
.venv/bin/python evaluate_models.py
```

The script uses scikit-learn's built-in breast cancer dataset. It keeps preprocessing inside a `Pipeline`, which prevents test data from influencing the scaler during training.

## Metrics

- **Accuracy:** fraction of predictions that are correct.
- **Precision:** among predicted positives, how many are actually positive.
- **Recall:** among actual positives, how many were found.
- **F1-score:** harmonic mean of precision and recall.
- **Confusion matrix:** counts true positives, true negatives, false positives, and false negatives.
- **MSE:** average squared difference between regression predictions and actual values.

## Interpretation

A model with high training accuracy but noticeably lower test accuracy may be overfitting. Cross-validation evaluates the same model over several training/validation splits and reports the mean and variation of the scores.

The regression section is a metric demonstration using a continuous target derived from the dataset; it is included to practice MSE and should not be treated as a meaningful real-world prediction task.

## Learning Summary

Today I learned that choosing a model requires more than checking one accuracy number. Different metrics expose different errors, confusion matrices show the kind of mistakes being made, and cross-validation gives a more stable estimate than one train/test split.

## Completion Checklist

- [x] Logistic Regression
- [x] Decision Tree
- [x] Random Forest
- [x] K-Nearest Neighbors
- [x] Accuracy, precision, recall, and F1-score
- [x] Confusion matrices
- [x] Mean squared error
- [x] Overfitting comparison
- [x] Cross-validation
