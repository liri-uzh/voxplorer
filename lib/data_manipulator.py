"""
This module contains functions for manipulating data.
If feature extraction mode was selected, the data is split into features and metadata variables;
It is then saved in the selected output directory and sent to the embedding projector from the main script.

If visualizer mode was selected, then the function is used to create a selection partition if the user has selected
to do so.
"""

import os
import json
import pandas as pd

# TODO: add the option to select metavars and option to select which metavariable to use for selction


def filter_selection(input_file: str, output_file: str) -> None:
    """
    Filters the selection partition from the data and saves it in the same folder 
    with the addition of '_selection' to the filename.

    Parameters:
    ----------- 
    output_file: str
        Path to the output file.
    """
    # Load the data
    if os.path.exists(input_file):
        basename = os.path.basename(input_file)
        filename, _ = os.path.splitext(basename)
        if os.path.isfile(os.path.join(filename + '_dtypes.json')):
            with open(os.path.join(filename + '_dtypes.json'), 'r') as f:
                dtypes = json.load(f)   # json dtypes
            dtypes = {k: pd.api.types.pandas_dtype(
                v) for k, v in dtypes.items()}  # pandas readable dtypes
            df = pd.read_csv(input_file, dtype=dtypes)
        else:
            df = pd.read_csv(input_file)
    else:
        raise FileNotFoundError("File not found.")

    # Filter the selection partition
    df = df[df['selection'] == 1]

    # Save the selection partition
    if not os.path.exists(os.path.dirname(output_file)):
        os.makedirs(os.path.dirname(output_file))
    output_filename = os.path.splitext(output_file)[0]
    df.to_csv(output_filename + '.csv', index=False)
    df.dtypes.apply(lambda x: x.name).to_json(output_filename + '_dtypes.json')


def extract_metadata(filename: str, metavars: list, separator: str = '_', ) -> dict:
    """
    Extracts metadata variables from the filename.

    Parameters:
    -----------
    filename: str
        Filename.
    metavars: list
        List of metadata variable names. If '-' then it is skipped.
    separator: str
        Separator character between metadata variables.

    Returns:
    --------
    dict
        Dictionary of metadata variables.
    """
    # Create empty dictionary
    metadata_dict = {}

    # Extract metadata from filename
    basename = os.path.basename(filename)
    metadata = os.path.splitext(basename)[0]
    metadata = metadata.split(separator)

    # append to dictionary a dictionary of metadata
    metadata_dict['filename'] = [basename]
    for j, var in enumerate(metavars):
        if var == '-':
            continue
        metadata_dict[var] = [metadata[j]]

    return metadata_dict
