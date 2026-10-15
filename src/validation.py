import pandas as pd

from sklearn.model_selection import (
    StratifiedKFold,
    cross_validate
)
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score
)

from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer

from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from sklearn.model_selection import GridSearchCV
def get_cv_models():
    """
    Create leakage-safe TF-IDF pipelines
    for cross-validation.
    """

    tfidf_params = {
        "max_features": 10000,
        "ngram_range": (1, 2),
        "min_df": 2,
        "max_df": 0.95,
        "sublinear_tf": True
    }

    models = {
        "Logistic Regression": Pipeline([
            (
                "tfidf",
                TfidfVectorizer(**tfidf_params)
            ),
            (
                "classifier",
                LogisticRegression(
                    max_iter=1000,
                    random_state=42
                )
            )
        ]),

        "Linear SVM": Pipeline([
            (
                "tfidf",
                TfidfVectorizer(**tfidf_params)
            ),
            (
                "classifier",
                LinearSVC(
                    random_state=42
                )
            )
        ]),

        "Multinomial Naive Bayes": Pipeline([
            (
                "tfidf",
                TfidfVectorizer(**tfidf_params)
            ),
            (
                "classifier",
                MultinomialNB()
            )
        ])
    }

    return models
def cross_validate_models(
    texts,
    targets,
    n_splits=5
):
    """
    Perform stratified cross-validation.

    TF-IDF is fitted independently inside
    each fold to prevent data leakage.
    """

    cv = StratifiedKFold(
        n_splits=n_splits,
        shuffle=True,
        random_state=42
    )

    scoring = {
        "accuracy": "accuracy",
        "precision": "precision",
        "recall": "recall",
        "f1": "f1",
        "roc_auc": "roc_auc"
    }

    models = get_cv_models()

    results = []

    for name, model in models.items():

        print(f"Cross-validating {name}...")

        scores = cross_validate(
            model,
            texts,
            targets,
            cv=cv,
            scoring=scoring,
            n_jobs=-1
        )

        results.append({
            "Model": name,

            "CV Accuracy":
                scores["test_accuracy"].mean(),

            "CV Precision":
                scores["test_precision"].mean(),

            "CV Recall":
                scores["test_recall"].mean(),

            "CV F1":
                scores["test_f1"].mean(),

            "CV ROC-AUC":
                scores["test_roc_auc"].mean(),

            "F1 Std":
                scores["test_f1"].std()
        })

    return pd.DataFrame(results)
def tune_logistic_regression(
    texts,
    targets
):
    """
    Tune Logistic Regression and TF-IDF parameters
    using leakage-safe 5-fold GridSearchCV.
    """

    pipeline = Pipeline([
        (
            "tfidf",
            TfidfVectorizer(
                max_features=10000,
                min_df=2,
                max_df=0.95,
                sublinear_tf=True
            )
        ),
        (
            "classifier",
            LogisticRegression(
                max_iter=1000,
                random_state=42
            )
        )
    ])

    param_grid = {
        "tfidf__ngram_range": [
            (1, 1),
            (1, 2)
        ],

        "classifier__C": [
            0.5,
            1.0,
            2.0
        ],

        "classifier__class_weight": [
            None,
            "balanced"
        ]
    }

    cv = StratifiedKFold(
        n_splits=5,
        shuffle=True,
        random_state=42
    )

    grid_search = GridSearchCV(
        estimator=pipeline,
        param_grid=param_grid,
        scoring="f1",
        cv=cv,
        n_jobs=-1,
        verbose=1,
        return_train_score=True
    )

    grid_search.fit(
        texts,
        targets
    )

    return grid_search
def tune_naive_bayes(
    texts,
    targets
):
    """
    Tune Multinomial Naive Bayes and TF-IDF
    parameters using 5-fold GridSearchCV.
    """

    pipeline = Pipeline([
        (
            "tfidf",
            TfidfVectorizer(
                max_features=10000,
                min_df=2,
                max_df=0.95,
                sublinear_tf=True
            )
        ),
        (
            "classifier",
            MultinomialNB()
        )
    ])

    param_grid = {
        "tfidf__ngram_range": [
            (1, 1),
            (1, 2)
        ],

        "classifier__alpha": [
            0.1,
            0.5,
            1.0,
            2.0
        ],

        "classifier__fit_prior": [
            True,
            False
        ]
    }

    cv = StratifiedKFold(
        n_splits=5,
        shuffle=True,
        random_state=42
    )

    grid_search = GridSearchCV(
        estimator=pipeline,
        param_grid=param_grid,
        scoring="f1",
        cv=cv,
        n_jobs=-1,
        verbose=1,
        return_train_score=True
    )

    grid_search.fit(
        texts,
        targets
    )

    return grid_search
def tune_linear_svm(
    texts,
    targets
):
    """
    Tune Linear SVM and TF-IDF parameters
    using leakage-safe 5-fold GridSearchCV.
    """

    pipeline = Pipeline([
        (
            "tfidf",
            TfidfVectorizer(
                max_features=10000,
                min_df=2,
                max_df=0.95,
                sublinear_tf=True
            )
        ),
        (
            "classifier",
            LinearSVC(
                random_state=42
            )
        )
    ])

    param_grid = {
        "tfidf__ngram_range": [
            (1, 1),
            (1, 2)
        ],

        "classifier__C": [
            0.1,
            0.5,
            1.0,
            2.0
        ],

        "classifier__class_weight": [
            None,
            "balanced"
        ]
    }

    cv = StratifiedKFold(
        n_splits=5,
        shuffle=True,
        random_state=42
    )

    grid_search = GridSearchCV(
        estimator=pipeline,
        param_grid=param_grid,
        scoring="f1",
        cv=cv,
        n_jobs=-1,
        verbose=1,
        return_train_score=True
    )

    grid_search.fit(
        texts,
        targets
    )

    return grid_search
def create_tuning_comparison(
    lr_grid,
    nb_grid,
    svm_grid
):
    """
    Create a comparison table for the best
    hyperparameter combination of each model.
    """

    models = {
        "Logistic Regression": lr_grid,
        "Multinomial Naive Bayes": nb_grid,
        "Linear SVM": svm_grid
    }

    results = []

    for name, grid in models.items():

        best_index = grid.best_index_

        train_f1 = grid.cv_results_[
            "mean_train_score"
        ][best_index]

        validation_f1 = grid.cv_results_[
            "mean_test_score"
        ][best_index]

        validation_std = grid.cv_results_[
            "std_test_score"
        ][best_index]

        results.append({
            "Model": name,
            "Training F1": train_f1,
            "Validation F1": validation_f1,
            "Validation F1 Std": validation_std,
            "Train-Validation Gap":
                train_f1 - validation_f1
        })

    return pd.DataFrame(results)

def evaluate_tuned_models(
    lr_grid,
    nb_grid,
    svm_grid,
    test_texts,
    y_test
):
    """
    Evaluate the tuned models once on the
    untouched holdout test dataset.
    """

    models = {
        "Tuned Logistic Regression":
            lr_grid.best_estimator_,

        "Tuned Multinomial Naive Bayes":
            nb_grid.best_estimator_,

        "Tuned Linear SVM":
            svm_grid.best_estimator_
    }

    results = []

    for name, model in models.items():

        y_pred = model.predict(test_texts)

        # Probability or decision score for ROC-AUC
        if hasattr(model, "predict_proba"):
            y_score = model.predict_proba(
                test_texts
            )[:, 1]

        elif hasattr(model, "decision_function"):
            y_score = model.decision_function(
                test_texts
            )

        else:
            y_score = None

        result = {
            "Model": name,

            "Accuracy":
                accuracy_score(
                    y_test,
                    y_pred
                ),

            "Precision":
                precision_score(
                    y_test,
                    y_pred,
                    zero_division=0
                ),

            "Recall":
                recall_score(
                    y_test,
                    y_pred,
                    zero_division=0
                ),

            "F1 Score":
                f1_score(
                    y_test,
                    y_pred,
                    zero_division=0
                )
        }

        if y_score is not None:
            result["ROC-AUC"] = (
                roc_auc_score(
                    y_test,
                    y_score
                )
            )

        results.append(result)

    return pd.DataFrame(results)
def create_error_analysis(
    model,
    test_df
):
    """
    Identify false-positive and false-negative
    predictions for error analysis.
    """

    results = test_df.copy()

    results["predicted_target"] = model.predict(
        results["processed_text"]
    )

    false_positives = results[
        (results["target"] == 0)
        &
        (results["predicted_target"] == 1)
    ].copy()

    false_negatives = results[
        (results["target"] == 1)
        &
        (results["predicted_target"] == 0)
    ].copy()

    return false_positives, false_negatives