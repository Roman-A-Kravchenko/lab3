from __future__ import annotations

import json

import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split

from linear_models import (
    _classification_metrics,
    compare_classical_models,
    train_linear_regression,
    train_logistic_classifier,
)
from grader_contracts.linear_models import ClassificationInput, RegressionInput


def main() -> None:
    df_student = pd.read_csv("Student_Performance.csv")
    df_student["Extracurricular Activities"] = (
        df_student["Extracurricular Activities"].map({"Yes": 1, "No": 0})
    )

    X_reg = df_student.drop(columns=["Performance Index"]).values
    y_reg = df_student["Performance Index"].values

    X_reg_train, X_reg_test, y_reg_train, y_reg_test = train_test_split(
        X_reg, y_reg, test_size=0.2, random_state=42
    )

    reg_input = RegressionInput(
        train_features=X_reg_train,
        train_target=y_reg_train,
        test_features=X_reg_test,
        test_target=y_reg_test,
        learning_rate=0.1,
        epochs=1000,
    )
    reg_result = train_linear_regression(reg_input)
    test_mse = float(mean_squared_error(y_reg_test, reg_result.predictions))

    df_german = pd.read_csv("german.csv", sep=";")
    y_german = df_german["Creditability"].values
    X_german = df_german.drop(columns=["Creditability"]).values

    X_g_train, X_g_test, y_g_train, y_g_test = train_test_split(
        X_german, y_german, test_size=0.2, random_state=42
    )

    clf_input = ClassificationInput(
        train_features=X_g_train,
        train_target=y_g_train,
        test_features=X_g_test,
        test_target=y_g_test,
        learning_rate=0.1,
        epochs=1000,
    )

    clf_result = train_logistic_classifier(clf_input)
    own_metrics = _classification_metrics(y_g_test, clf_result.probabilities)

    comparison = compare_classical_models(clf_input)

    results = {
        "linear_regression": {
            "test_mse": round(test_mse, 4),
        },
        "classification": {
            "own_logistic": {
                "roc_auc": round(own_metrics.roc_auc, 4),
                "accuracy": round(own_metrics.accuracy, 4),
                "precision": round(own_metrics.precision, 4),
                "recall": round(own_metrics.recall, 4),
                "f1": round(own_metrics.f1, 4),
            },
            "logistic_regression": {
                "roc_auc": round(comparison.logistic_regression.roc_auc, 4),
                "accuracy": round(comparison.logistic_regression.accuracy, 4),
                "precision": round(comparison.logistic_regression.precision, 4),
                "recall": round(comparison.logistic_regression.recall, 4),
                "f1": round(comparison.logistic_regression.f1, 4),
            },
            "decision_tree": {
                "roc_auc": round(comparison.decision_tree.roc_auc, 4),
                "accuracy": round(comparison.decision_tree.accuracy, 4),
                "precision": round(comparison.decision_tree.precision, 4),
                "recall": round(comparison.decision_tree.recall, 4),
                "f1": round(comparison.decision_tree.f1, 4),
            },
            "knn": {
                "roc_auc": round(comparison.knn.roc_auc, 4),
                "accuracy": round(comparison.knn.accuracy, 4),
                "precision": round(comparison.knn.precision, 4),
                "recall": round(comparison.knn.recall, 4),
                "f1": round(comparison.knn.f1, 4),
            },
        },
    }

    with open("experiment_results.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print("experiment_results.json создан")
    print(json.dumps(results, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
