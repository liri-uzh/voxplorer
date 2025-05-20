"""Data loading module for embedding projector.
Data is loaded using a path and then divided into the variables and features.
Also contains a function to filter the data by a metavariable.

----------
Alessandro De Luca - 09/2023
"""
import os
import json
from tkinter import messagebox
import numpy as np
import pandas as pd


def load_data(path_to_data: str, metavars: list = None, use_json_dtypes: bool = False):
    """
    Loads a table given a path. Infers that correct separator character.

    Parameters:
    -----------
    path_to_data: str
        Path to the data table.
    metavars: list
        List of metadata variable names. Used to select the dtype of metadata variables.
        If None, normal inference of dytpes is used.

    Returns:
    --------
    pd.DataFrame
        Data table.
    """
    # Find the appropriate separator character
    with open(path_to_data, 'r') as f:
        line = f.readline()
        if len(line.split(',')) > 1:
            sep = ','
        elif len(line.split('\t')) > 1:
            sep = '\t'
        elif len(line.split(' ')) > 1:
            sep = ' '
        elif len(line.split(';')) > 1:
            sep = ';'
        else:
            sep = input("What is the separator character for the data?")

    try:
        if use_json_dtypes:
            filename = os.path.splitext(path_to_data)[0]
            try:
                with open(os.path.join(filename + '_dtypes.json'), 'r') as f:
                    dtypes = json.load(f)
                dtypes = {k: pd.api.types.pandas_dtype(
                    v) for k, v in dtypes.items()}
            except FileNotFoundError:
                messagebox.showerror(
                    "Error", "Cannot find dtypes file. This should be saved in the same folder as the data file with the extension _dtypes.json.")
                return
            # TODO: implement dtypes to metavars if metavars is None
            df = pd.read_csv(path_to_data, sep=sep, dtype=dtypes)

        elif metavars is not None:
            dtypes = {k: pd.api.types.pandas_dtype(
                'category') for k in metavars}

            df = pd.read_csv(path_to_data, sep=sep, dtype=dtypes)
        else:
            df = pd.read_csv(path_to_data, sep=sep)
    except:
        UnicodeDecodeError(
            "Cannot read the file")

    return df


def split_data(df: pd.DataFrame, features: list, metavars: list, add_selection_column: bool = False):
    """
    Splits the data into features and metadata.

    Parameters:
    -----------
    df: pd.DataFrame
        Data table.
    features: list
        List of features variable names or indices list.
        If None, all are taken except those listed in metavars.
        If 'all', all are taken.
    metavars: list
        List of metadata variable names.
        If None, metadata variables are excluded.
        If 'infer', metadata variables are inferred as the ones not selected as features.
    add_selection_column: bool
        Whether to add a selection column to the metadata variables.


    Returns:
    --------
    X : np.ndarray
        Features array.
    metadata : np.ndarray
        Metadata values (if metavars is not None).
    metavars : list
        Metadata variable names (if metavars is not None)

    Caveats:
    --------
    If both metavars and features are None, all of the data is taken as features.
    If features is None and
    """

    # Remove an NA values
    if df.isna().any(axis=None):
        print("Warning: Data contains NAs\nremoving...")
        df = df.dropna()

    if add_selection_column:
        df['selection'] = np.repeat(0, df.shape[0])
        metavars.append('selection')

    # Feature selection and metadata variable selection
    # Do all checks on features and metavars to understand what needs to go where
    if (features is None and metavars is None) or features == 'all':
        X = df.to_numpy()
        return X, None, None

    if features is None and metavars is not None:
        if type(metavars) is str:
            if metavars == 'infer':
                raise ValueError("Cannot infer metavars when features is None")
        elif type(metavars[0]) is int:
            metavars = df.columns[metavars].to_list()

        X = df.loc[:, ~df.columns.isin(metavars)].astype(np.float64).to_numpy()
        Y = df.loc[:, metavars].astype(str).to_numpy()
        return X, Y, metavars

    if features is not None and metavars is None:
        if type(features[0]) is int:
            features = df.columns[features]

        X = df.loc[:, features].astype(np.float64).to_numpy()
        return X, None, None

    if features is not None and metavars is not None:
        if metavars == 'infer':
            if type(features[0]) is int:
                features = df.columns[features]

            X = df.loc[:, features].astype(np.float64).to_numpy()
            Y = df.loc[:, ~df.columns.isin(features)]
            metavars = Y.columns.to_list()
            Y = Y.astype(str).to_numpy()
            return X, Y, metavars
        else:
            if type(features[0]) is int:
                features = df.columns[features]
            if type(metavars[0]) is int:
                metavars = df.columns[metavars].to_list()

            X = df.loc[:, features].astype(np.float64).to_numpy()
            Y = df.loc[:, metavars].astype(str).to_numpy()
            return X, Y, metavars


def filter_selection(df: pd.DataFrame, output_file: str, metavar: str = 'selection', metavar_filter: str = '1') -> None:
    """
    Filters the selection partition from the data and saves it in the same folder 
    with the addition of '_selection' to the filename.

    Parameters:
    ----------- 
    df: pd.DataFrame
        Data table.
    output_file: str
        Path to the output file.
    metavar: str
        Name of the metadata variable to filter.
    metavar_filter: str
        Value of the metadata variable to filter.
    """
    # Filter the selection partition
    df_filtered = df[df[metavar] == metavar_filter]

    if df_filtered.shape[0] == 0:
        # Try using int
        df_filtered = df[df[metavar] == int(metavar_filter)]

    if df_filtered.shape[0] == 0:
        messagebox.showerror(
            "Error", "Filtering leads to empty dataframe.")
        return

    # Save the selection partition
    if not os.path.exists(os.path.dirname(output_file)):
        os.makedirs(os.path.dirname(output_file))
    output_filename = os.path.splitext(output_file)[0]
    df_filtered.to_csv(output_filename + '.csv', index=False)
    df_filtered.dtypes.apply(lambda x: x.name).to_json(
        output_filename + '_dtypes.json')
