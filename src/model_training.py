from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier

def get_models(random_state=42):
    """
    Create the machine-learning models used for
    disaster tweet classification.

    Returns
    -------
    dict
        Dictionary containing model names and objects.
    """

    models = {
        "Logistic Regression": LogisticRegression(
            max_iter=1000,
            random_state=random_state
        ),

        "Linear SVM": LinearSVC(
            random_state=random_state
        ),

        "Random Forest": RandomForestClassifier(
            n_estimators=200,
            random_state=random_state,
            n_jobs=-1
        )
    }

    return models
def train_models(X_train, y_train):
    """
    Train multiple classification models.

    Parameters
    ----------
    X_train : feature matrix
        Combined training features.

    y_train : pandas Series
        Training target labels.

    Returns
    -------
    dict
        Dictionary containing trained models.
    """

    models = get_models()

    trained_models = {}

    for name, model in models.items():

        print(f"Training {name}...")

        model.fit(
            X_train,
            y_train
        )

        trained_models[name] = model

        print(f"{name} training completed.\n")

    return trained_models
def train_naive_bayes(
    X_train_tfidf,
    y_train
):
    """
    Train Multinomial Naive Bayes using TF-IDF features.

    TF-IDF is used separately because MultinomialNB
    requires non-negative input features.
    """

    model = MultinomialNB()

    model.fit(
        X_train_tfidf,
        y_train
    )

    return model