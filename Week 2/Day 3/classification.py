"""Train and evaluate a baseline classification model."""

from sklearn.datasets import load_breast_cancer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


def main():
    dataset = load_breast_cancer()
    features_train, features_test, labels_train, labels_test = train_test_split(
        dataset.data,
        dataset.target,
        test_size=0.2,
        random_state=42,
        stratify=dataset.target,
    )

    model = Pipeline(
        [
            ("scaler", StandardScaler()),
            ("classifier", LogisticRegression(max_iter=2000)),
        ]
    )
    model.fit(features_train, labels_train)
    predictions = model.predict(features_test)

    accuracy = accuracy_score(labels_test, predictions)
    print(f"Samples: {len(dataset.data)}")
    print(f"Features: {dataset.data.shape[1]}")
    print(f"Training samples: {len(features_train)}")
    print(f"Testing samples: {len(features_test)}")
    print(f"Accuracy: {accuracy:.3f}")
    print("\nClassification report:")
    print(classification_report(labels_test, predictions, target_names=dataset.target_names))


if __name__ == "__main__":
    main()
