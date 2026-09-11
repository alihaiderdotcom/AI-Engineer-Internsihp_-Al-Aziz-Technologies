"""End-to-end ML project: data cleaning, EDA, preprocessing, training, and evaluation."""

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier


BASE_DIR = Path(__file__).parent
RESULTS_FILE = BASE_DIR / "iris_results.txt"


def phase1_load_and_explore():
    """Load data and perform initial exploration."""
    dataset = load_iris()
    df = pd.DataFrame(dataset.data, columns=dataset.feature_names)
    df["target"] = dataset.target
    df["target_name"] = df["target"].map({
        0: "setosa",
        1: "versicolor",
        2: "virginica",
    })
    
    print("=" * 60)
    print("PHASE 1: DATA EXPLORATION")
    print("=" * 60)
    print(f"\nDataset shape: {df.shape}")
    print(f"Missing values:\n{df.isnull().sum()}")
    print(f"Duplicate rows: {df.duplicated().sum()}")
    print(f"\nFeature statistics:\n{df.drop('target_name', axis=1).describe().round(2)}")
    print(f"\nClass distribution:\n{df['target_name'].value_counts()}")
    print(f"\nCorrelation matrix:\n{df.drop('target_name', axis=1).corr().round(2)}")
    
    return df, dataset


def phase2_preprocessing(df, dataset):
    """Prepare data for model training."""
    print("\n" + "=" * 60)
    print("PHASE 2: PREPROCESSING")
    print("=" * 60)
    
    X = df.drop(['target', 'target_name'], axis=1)
    y = df['target']
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    print(f"\nTraining set: {X_train.shape[0]} samples")
    print(f"Test set: {X_test.shape[0]} samples")
    print(f"Features: {X_train.shape[1]}")
    
    return X_train, X_test, y_train, y_test, dataset.target_names


def phase3_model_training(X_train, X_test, y_train, y_test, target_names):
    """Train and evaluate multiple classifiers."""
    print("\n" + "=" * 60)
    print("PHASE 3: MODEL TRAINING & EVALUATION")
    print("=" * 60)
    
    models = {
        "Logistic Regression": Pipeline([
            ("scaler", StandardScaler()),
            ("model", LogisticRegression(max_iter=200, random_state=42))
        ]),
        "Decision Tree": DecisionTreeClassifier(max_depth=5, random_state=42),
        "Random Forest": RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42),
    }
    
    results = {}
    for name, model in models.items():
        model.fit(X_train, y_train)
        train_pred = model.predict(X_train)
        test_pred = model.predict(X_test)
        
        train_acc = accuracy_score(y_train, train_pred)
        test_acc = accuracy_score(y_test, test_pred)
        
        results[name] = {
            "model": model,
            "train_accuracy": train_acc,
            "test_accuracy": test_acc,
            "predictions": test_pred,
            "report": classification_report(y_test, test_pred, target_names=target_names),
            "confusion_matrix": confusion_matrix(y_test, test_pred).tolist(),
        }
        
        print(f"\n{name}:")
        print(f"  Train accuracy: {train_acc:.3f}")
        print(f"  Test accuracy:  {test_acc:.3f}")
        print(f"  Overfitting gap: {abs(train_acc - test_acc):.3f}")
    
    return results


def phase4_report(results):
    """Generate and save final report."""
    print("\n" + "=" * 60)
    print("PHASE 4: FINAL REPORT")
    print("=" * 60)
    
    best_model_name = max(results, key=lambda k: results[k]["test_accuracy"])
    best_result = results[best_model_name]
    
    print(f"\nBest Model: {best_model_name}")
    print(f"Test Accuracy: {best_result['test_accuracy']:.3f}")
    print(f"\nClassification Report:\n{best_result['report']}")
    
    report_content = f"""WEEK 2 ML PROJECT REPORT
========================

Project: Iris Flower Classification

SUMMARY
-------
Best performing model: {best_model_name}
Test accuracy: {best_result['test_accuracy']:.3f}

MODEL COMPARISON
----------------
"""
    
    for name, result in results.items():
        report_content += f"\n{name}:\n"
        report_content += f"  Train accuracy: {result['train_accuracy']:.3f}\n"
        report_content += f"  Test accuracy:  {result['test_accuracy']:.3f}\n"
        report_content += f"  Confusion matrix: {result['confusion_matrix']}\n"
    
    report_content += f"\nRECOMMENDATION\n"
    report_content += f"Use {best_model_name} for production deployment.\n"
    
    RESULTS_FILE.write_text(report_content, encoding="utf-8")
    print(f"\nReport saved to {RESULTS_FILE}")


def main():
    df, dataset = phase1_load_and_explore()
    X_train, X_test, y_train, y_test, target_names = phase2_preprocessing(df, dataset)
    results = phase3_model_training(X_train, X_test, y_train, y_test, target_names)
    phase4_report(results)
    
    print("\n" + "=" * 60)
    print("WEEK 2 PROJECT COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    main()
