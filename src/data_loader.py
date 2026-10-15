"""
Data Loading Module
-------------------
This module contains reusable functions for loading
and inspecting the Disaster Tweet Classification dataset.
"""

from pathlib import Path
import pandas as pd


def load_data(file_path):
    """
    Load the disaster tweet dataset from an Excel file.

    Parameters
    ----------
    file_path : str or pathlib.Path
        Path to the Excel dataset.

    Returns
    -------
    pandas.DataFrame
        Loaded dataset.
    """

    file_path = Path(file_path)

    # Check whether file exists
    if not file_path.exists():
        raise FileNotFoundError(
            f"Dataset not found: {file_path}"
        )

    # Check file extension
    if file_path.suffix.lower() not in [".xlsx", ".xls"]:
        raise ValueError(
            "Dataset must be an Excel file."
        )

    # Load Excel dataset
    df = pd.read_excel(file_path)

    return df