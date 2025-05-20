"""Module to load data from a database.

This module contains functions to load data from a database table and update the column type of a DataFrame.

Functions
---------
load_data(table_path, sep=None)
    Load data from a database table.
update_column_dtype(data, column_name, dtype)
    Update the column type of a DataFrame.
"""

import pandas as pd


def load_data(table_path, sep: str = None):
    """Load data from a database table.

    Parameters
    ----------
    table_path : str
        Path to the table in the database.
    sep : str
        Separator of the table. Default is None. If None, the function will try to infer the separator.

    Returns
    -------
    data : pandas.DataFrame
        Data from the table.
    or Error:
    - FileNotFoundError: Could not find the file. Please provide a valid file.
    - ValueError: Could not infer the separator. Please provide the separator.
    - UnicodeDecodeError: Could not read the file. Please provide a valid file.
    """
    try:
        if sep is None:
            with open(table_path, "r") as f:
                line = f.readline()
                if len(line.split(",")) > 1:
                    sep = ","  # CSV
                elif len(line.split("\t")) > 1:
                    sep = "\t"  # TSV
                elif len(line.split(";")) > 1:
                    sep = ";"  # Semicolon
                elif len(line.split(" ")) > 1:
                    sep = " "  # Space
                else:
                    return ValueError(
                        "SeparatorError: Could not infer the separator. Please provide the separator."
                    )
    except FileNotFoundError:
        raise FileNotFoundError("Could not find the file. Please provide a valid file.")

    try:
        df = pd.read_csv(table_path, sep=sep)
    except UnicodeDecodeError:
        return UnicodeDecodeError(
            "Could not read the file. Please provide a valid file."
        )

    return df


def update_column_dtype(data, column_name, dtype):
    """Update the column type of a DataFrame.

    Parameters
    ----------
    data : pandas.DataFrame
        DataFrame to update.
    column_name : str
        Name of the column to update.
    dtype : str
        New type of the column. (Options: int, float, str, bool, category)

    Returns
    -------
    data : pandas.DataFrame
        DataFrame with the updated column type.
    """
    data[column_name] = data[column_name].astype(pd.api.types.pandas_dtype(dtype))
    return data
