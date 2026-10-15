import re
import pandas as pd


from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer


def handle_missing_values(df):
    """
    Handle missing values in the disaster tweet dataset.
    """

    df = df.copy()

    df["keyword"] = df["keyword"].fillna("unknown")
    df["location"] = df["location"].fillna("unknown")

    df = df.dropna(subset=["text"])
    df = df.dropna(subset=["target"])

    return df


def remove_duplicates(df):
    """
    Remove exact duplicate rows from the dataset.
    """

    df = df.copy()

    df = df.drop_duplicates()

    df = df.reset_index(drop=True)

    return df


def clean_text(text):
    """
    Clean raw tweet text for NLP processing.
    """

    # Convert to string and lowercase
    text = str(text).lower()

    # Replace HTML encoded ampersand
    text = text.replace("&amp;", " and ")

    # Remove URLs
    text = re.sub(
        r"https?://\S+|www\.\S+",
        " ",
        text
    )

    # Remove HTML tags
    text = re.sub(
        r"<.*?>",
        " ",
        text
    )

    # Remove Twitter mentions
    text = re.sub(
        r"@\w+",
        " ",
        text
    )

    # Remove # but keep hashtag word
    text = re.sub(
        r"#(\w+)",
        r"\1",
        text
    )

    # Remove numbers, punctuation and special characters
    text = re.sub(
        r"[^a-z\s]",
        " ",
        text
    )

    # Remove extra whitespace
    text = re.sub(
        r"\s+",
        " ",
        text
    ).strip()

    return text

def preprocess_text(text):
    """
    Perform NLP preprocessing on cleaned tweet text.

    Steps:
    1. Tokenize text
    2. Remove English stopwords
    3. Lemmatize words
    4. Join tokens back into text

    Parameters
    ----------
    text : str
        Cleaned tweet text.

    Returns
    -------
    str
        Fully preprocessed tweet text.
    """

    # Initialize stopwords
    stop_words = set(stopwords.words("english"))

    # Initialize lemmatizer
    lemmatizer = WordNetLemmatizer()

    # Tokenization
    tokens = text.split()

    # Remove stopwords
    tokens = [
        word
        for word in tokens
        if word not in stop_words
    ]

    # Lemmatization
    tokens = [
        lemmatizer.lemmatize(word)
        for word in tokens
    ]

    # Convert tokens back into text
    processed_text = " ".join(tokens)

    return processed_text

def preprocess_data(df):
    """
    Run the complete preprocessing pipeline
    on the disaster tweet dataset.

    Parameters
    ----------
    df : pandas.DataFrame
        Raw disaster tweet dataset.

    Returns
    -------
    pandas.DataFrame
        Fully preprocessed dataset.
    """

    # Handle missing values
    df = handle_missing_values(df)

    # Remove exact duplicates
    df = remove_duplicates(df)

    # Basic text cleaning
    df["clean_text"] = (
        df["text"]
        .apply(clean_text)
    )

    # NLP preprocessing
    df["processed_text"] = (
        df["clean_text"]
        .apply(preprocess_text)
    )

    # Remove rows that became empty
    df = df[
        df["processed_text"]
        .str.strip()
        .ne("")
    ]

    # Reset index
    df = df.reset_index(drop=True)

    return df

