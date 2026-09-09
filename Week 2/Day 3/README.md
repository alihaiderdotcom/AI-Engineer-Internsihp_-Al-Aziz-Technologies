# Week 2, Day 3 - Machine Learning Fundamentals

**Focus:** Supervised learning, classification, features, labels, preprocessing, and train/test split.

## Learning Objectives

- Distinguish AI, machine learning, and deep learning.
- Identify features and a target label.
- Split data into training and testing sets.
- Scale numeric features without leaking test information.
- Train a classification model with scikit-learn.
- Use predictions and accuracy to evaluate a first model.

## Run

```bash
.venv/bin/python classification.py
```

The exercise uses scikit-learn's built-in breast cancer dataset, so no external dataset download is required. The model is a baseline learning exercise and is not a medical diagnostic system.

## Workflow

1. Load the dataset.
2. Separate features `X` from target labels `y`.
3. Create stratified training and test sets.
4. Fit a `StandardScaler` on the training data only.
5. Train a logistic regression classifier.
6. Predict on unseen test data and report accuracy.

## Learning Summary

Today I learned that supervised learning uses labeled examples to learn a mapping from features to a target. A train/test split estimates how well the model generalizes. Preprocessing must be fitted on training data only to avoid leaking information from the test set.

## Completion Checklist

- [x] AI vs. ML vs. deep learning overview
- [x] Supervised and unsupervised learning concepts
- [x] Classification, regression, and clustering overview
- [x] Features and labels
- [x] Training and testing data
- [x] Train/test split
- [x] Data preprocessing and feature scaling
- [x] Baseline classification model
