from pathlib import Path
import joblib


def save_model(model, file_path):
    """
    Save a trained model or pipeline to disk.
    """

    file_path = Path(file_path)

    file_path.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    joblib.dump(
        model,
        file_path
    )

    return file_path


def load_model(file_path):
    """
    Load a saved model or pipeline from disk.
    """

    file_path = Path(file_path)

    if not file_path.exists():
        raise FileNotFoundError(
            f"Model not found: {file_path}"
        )

    return joblib.load(file_path)