"""Compare classification models and demonstrate evaluation metrics."""

import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    mean_squared_error,
    precision_score,
    recall_score,
)
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier


RANDOM_STATE = 42


def build_models():
    return {
        "Logistic Regression": Pipeline(
            [("scaler", StandardScaler()), ("model", LogisticRegression(max_iter=2000))]
        ),
        "Decision Tree": DecisionTreeClassifier(max_depth=4, random_state=RANDOM_STATE),
        "Random Forest": RandomForestClassifier(
            n_estimators=150, max_depth=6, random_state=RANDOM_STATE
        ),
        "K-Nearest Neighbors": Pipeline(
            [("scaler", StandardScaler()), ("model", KNeighborsClassifier(n_neighbors=5))]
        ),
    }


def evaluate_classifiers(features_train, features_test, labels_train, labels_test):
    results = {}
    for name, model in build_models().items():
        model.fit(features_train, labels_train)
        train_predictions = model.predict(features_train)
        test_predictions = model.predict(features_test)
        results[name] = {
            "train_accuracy": accuracy_score(labels_train, train_predictions),
            "test_accuracy": accuracy_score(labels_test, test_predictions),
            "precision": precision_score(labels_test, test_predictions),
            "recall": recall_score(labels_test, test_predictions),
            "f1": f1_score(labels_test, test_predictions),
            "confusion_matrix": confusion_matrix(labels_test, test_predictions).tolist(),
        }
    return results


def evaluate_regression(features, labels):
    regression_target = features[:, 0]
    regression_features = features[:, 1:]
    train_x, test_x, train_y, test_y = train_test_split(
        regression_features, regression_target, test_size=0.2, random_state=RANDOM_STATE
    )
    models = {
        "Linear Regression": LinearRegression(),
        "Mean Baseline": DummyRegressor(strategy="mean"),
    }
    return {
        name: mean_squared_error(test_y, model.fit(train_x, train_y).predict(test_x))
        for name, model in models.items()
    }


def main():
    dataset = load_breast_cancer()
    features_train, features_test, labels_train, labels_test = train_test_split(
        dataset.data,
        dataset.target,
        test_size=0.2,
        random_state=RANDOM_STATE,
        stratify=dataset.target,
    )

    results = evaluate_classifiers(features_train, features_test, labels_train, labels_test)
    print("Classification model comparison")
    for name, metrics in results.items():
        gap = metrics["train_accuracy"] - metrics["test_accuracy"]
        print(f"\n{name}")
        print(f"  train accuracy: {metrics['train_accuracy']:.3f}")
        print(f"  test accuracy:  {metrics['test_accuracy']:.3f}")
        print(f"  precision:      {metrics['precision']:.3f}")
        print(f"  recall:         {metrics['recall']:.3f}")
        print(f"  F1-score:       {metrics['f1']:.3f}")
        print(f"  train-test gap: {gap:.3f}")
        print(f"  confusion matrix: {metrics['confusion_matrix']}")

    cross_validation_model = build_models()["Logistic Regression"]
    cross_validation_scores = cross_val_score(
        cross_validation_model, dataset.data, dataset.target, cv=5, scoring="accuracy"
    )
    print("\n5-fold cross-validation")
    print(f"  scores: {np.round(cross_validation_scores, 3).tolist()}")
    print(f"  mean: {cross_validation_scores.mean():.3f}")
    print(f"  standard deviation: {cross_validation_scores.std():.3f}")

    print("\nRegression MSE demonstration")
    for name, mse in evaluate_regression(dataset.data, dataset.target).items():
        print(f"  {name}: {mse:.3f}")


if __name__ == "__main__":
    main()
