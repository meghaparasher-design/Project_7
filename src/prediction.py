from src.preprocessing import (
    clean_text,
    preprocess_text
)


def prepare_tweet(tweet):
    """
    Clean and preprocess a single tweet
    using the same preprocessing used
    during model training.
    """

    cleaned = clean_text(tweet)

    processed = preprocess_text(
        cleaned
    )

    return processed


def predict_tweet(model, tweet):
    """
    Predict whether a tweet represents
    a disaster or non-disaster.
    """

    if not isinstance(tweet, str):
        raise TypeError(
            "Tweet must be a string."
        )

    if not tweet.strip():
        raise ValueError(
            "Tweet cannot be empty."
        )

    processed_tweet = prepare_tweet(
        tweet
    )

    prediction = model.predict(
        [processed_tweet]
    )[0]

    label = (
        "Disaster"
        if prediction == 1
        else "Non-Disaster"
    )

    return {
        "original_text": tweet,
        "processed_text": processed_tweet,
        "prediction": int(prediction),
        "label": label
    }