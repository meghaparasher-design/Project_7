import re
import pandas as pd

from textblob import TextBlob
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler
from scipy.sparse import hstack, csr_matrix
NUMERIC_FEATURES = [
    "tweet_length",
    "word_count",
    "hashtag_count",
    "mention_count",
    "url_count",
    "exclamation_count",
    "question_count",
    "sentiment"
]
def create_text_features(df):
    """
    Create additional numerical features from tweet text.

    Parameters
    ----------
    df : pandas.DataFrame
        Preprocessed disaster tweet dataset.

    Returns
    -------
    pandas.DataFrame
        Dataset containing additional numerical features.
    """

    df = df.copy()

    # Number of characters in the original tweet
    df["tweet_length"] = (
        df["text"]
        .astype(str)
        .apply(len)
    )

    # Number of words in the original tweet
    df["word_count"] = (
        df["text"]
        .astype(str)
        .apply(
            lambda text: len(text.split())
        )
    )

    # Number of hashtags
    df["hashtag_count"] = (
        df["text"]
        .astype(str)
        .apply(
            lambda text: len(
                re.findall(r"#\w+", text)
            )
        )
    )

    # Number of user mentions
    df["mention_count"] = (
        df["text"]
        .astype(str)
        .apply(
            lambda text: len(
                re.findall(r"@\w+", text)
            )
        )
    )

    # Number of URLs
    df["url_count"] = (
        df["text"]
        .astype(str)
        .apply(
            lambda text: len(
                re.findall(
                    r"https?://\S+|www\.\S+",
                    text
                )
            )
        )
    )

    # Number of exclamation marks
    df["exclamation_count"] = (
        df["text"]
        .astype(str)
        .apply(
            lambda text: text.count("!")
        )
    )

    # Number of question marks
    df["question_count"] = (
        df["text"]
        .astype(str)
        .apply(
            lambda text: text.count("?")
        )
    )

    # Sentiment polarity
    df["sentiment"] = (
        df["text"]
        .astype(str)
        .apply(
            lambda text:
            TextBlob(text).sentiment.polarity
        )
    )

    return df
def split_data(
    df,
    test_size=0.20,
    random_state=42
):
    """
    Split the dataset into training and testing sets.

    Stratification preserves the target class distribution
    in both sets.
    """

    train_df, test_df = train_test_split(
        df,
        test_size=test_size,
        random_state=random_state,
        stratify=df["target"]
    )

    train_df = train_df.reset_index(drop=True)
    test_df = test_df.reset_index(drop=True)

    return train_df, test_df


def create_tfidf_features(
    train_text,
    test_text,
    max_features=10000
):
    """
    Convert processed tweet text into TF-IDF features.

    The vectorizer is fitted ONLY on training data
    to prevent data leakage.
    """

    vectorizer = TfidfVectorizer(
        max_features=max_features,
        ngram_range=(1, 2),
        min_df=2,
        max_df=0.95,
        sublinear_tf=True
    )

    # Learn vocabulary only from training data
    X_train_tfidf = vectorizer.fit_transform(
        train_text
    )

    # Transform test data using training vocabulary
    X_test_tfidf = vectorizer.transform(
        test_text
    )

    return (
        X_train_tfidf,
        X_test_tfidf,
        vectorizer
    )