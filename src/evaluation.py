import pandas as pd

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix,
    roc_auc_score,
        roc_curve,
        precision_recall_curve,
    average_precision_score
)
def calculate_metrics(
    model,
    X_test,
    y_test
):
    """
    Evaluate a classification model using common
    classification metrics.
    """

    y_pred = model.predict(X_test)

    metrics = {
        "Accuracy": accuracy_score(
            y_test,
            y_pred
        ),

        "Precision": precision_score(
            y_test,
            y_pred,
            zero_division=0
        ),

        "Recall": recall_score(
            y_test,
            y_pred,
            zero_division=0
        ),

        "F1 Score": f1_score(
            y_test,
            y_pred,
            zero_division=0
        )
    }

    return metrics

def evaluate_models(
    trained_models,
    naive_bayes_model,
    X_test,
    X_test_tfidf,
    y_test
):
    """
    Evaluate all trained models and return
    their performance metrics.
    """

    results = []

    # Models using combined features
    for name, model in trained_models.items():

        metrics = calculate_metrics(
            model,
            X_test,
            y_test
        )

        metrics["Model"] = name

        results.append(metrics)

    # Naive Bayes uses TF-IDF only
    nb_metrics = calculate_metrics(
        naive_bayes_model,
        X_test_tfidf,
        y_test
    )

    nb_metrics["Model"] = (
        "Multinomial Naive Bayes"
    )

    results.append(nb_metrics)

    results_df = pd.DataFrame(results)

    results_df = results_df[
        [
            "Model",
            "Accuracy",
            "Precision",
            "Recall",
            "F1 Score"
        ]
    ]

    return results_df
def get_classification_report(
    model,
    X_test,
    y_test
):
    """
    Generate a classification report.
    """

    y_pred = model.predict(X_test)

    report = classification_report(
        y_test,
        y_pred,
        target_names=[
            "Non-Disaster",
            "Disaster"
        ],
        zero_division=0
    )

    return report

def get_confusion_matrix(
    model,
    X_test,
    y_test
):
    """
    Generate the confusion matrix for a model.
    """

    y_pred = model.predict(X_test)

    cm = confusion_matrix(
        y_test,
        y_pred
    )

    return cm


import matplotlib.pyplot as plt
import seaborn as sns


def plot_confusion_matrix(
    model,
    X_test,
    y_test,
    model_name
):
    """
    Plot the confusion matrix for a classification model.
    """

    y_pred = model.predict(X_test)

    cm = confusion_matrix(
        y_test,
        y_pred
    )

    plt.figure(figsize=(6, 5))

    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=[
            "Non-Disaster",
            "Disaster"
        ],
        yticklabels=[
            "Non-Disaster",
            "Disaster"
        ]
    )

    plt.title(
        f"Confusion Matrix - {model_name}"
    )

    plt.xlabel("Predicted Label")
    plt.ylabel("Actual Label")

    plt.tight_layout()
    plt.show()
    
def get_model_scores(model, X_test):
    """
    Get continuous prediction scores for ROC-AUC evaluation.

    Uses predict_proba when available.
    Otherwise uses decision_function.
    """

    if hasattr(model, "predict_proba"):

        scores = model.predict_proba(
            X_test
        )[:, 1]

    elif hasattr(model, "decision_function"):

        scores = model.decision_function(
            X_test
        )

    else:

        raise ValueError(
            "Model does not support predict_proba "
            "or decision_function."
        )

    return scores
def calculate_roc_auc(
    model,
    X_test,
    y_test
):
    """
    Calculate ROC-AUC score for a classification model.
    """

    scores = get_model_scores(
        model,
        X_test
    )

    auc_score = roc_auc_score(
        y_test,
        scores
    )

    return auc_score
def calculate_average_precision(
    model,
    X_test,
    y_test
):
    """
    Calculate Average Precision (AP) score.

    AP summarizes the Precision-Recall curve
    into a single value.
    """

    scores = get_model_scores(
        model,
        X_test
    )

    ap_score = average_precision_score(
        y_test,
        scores
    )

    return ap_score
def get_precision_recall_data(
    model,
    X_test,
    y_test
):
    """
    Generate Precision-Recall curve data.
    """

    scores = get_model_scores(
        model,
        X_test
    )

    precision, recall, thresholds = (
        precision_recall_curve(
            y_test,
            scores
        )
    )

    ap_score = average_precision_score(
        y_test,
        scores
    )

    return precision, recall, ap_score