import os
import base64
import io
import dash_bootstrap_components as dbc
import polars as pl
from .feature_extraction import FeatureExtractor

ALLOWED_EXTENSIONS_AUDIO = {".wav", ".mp3", ".flac"}


def parse_table_contents(contents, filename):
    """Load data from a database table.

    Parameters
    ----------
    contents
        The parsed content.
    filename
        The filename.

    Returns
    -------
    data_table : pandas.DataFrame | None
        Data from the table or None.
    """
    contents_type, content_string = contents.split(",")

    # decode contents
    decoded = base64.b64decode(content_string)

    try:
        # generate file stream
        file_stream = io.BytesIO(decoded)

        ext = os.path.splitext(filename)[-1]
        if ext == ".csv":
            data_table = pl.read_csv(file_stream)
        elif ext == "tsv":
            data_table = pl.read_csv(file_stream, separator="\t")
        elif ext in (".xls", ".xlsb", ".xlsx"):
            data_table = pl.read_excel(file_stream)
        else:
            decode_exception = f"File {filename} could not be decoded."
            if ext == ".txt":
                decode_exception += (
                    "\nPlease modify the extension from '.txt' to "
                    + "'.csv' for comma-separated values tables or '.tsv' for "
                    + "tab-separated values tables."
                )
            return None, dbc.Alert(
                decode_exception,
                color="danger",
            )
    except Exception as e:
        return None, dbc.Alert(
            f"There was an error processing file {filename}: {e}",
            color="danger",
        )

    # Make JSON serialisable
    try:
        data_table = data_table.to_dicts()
    except Exception as e:
        return None, dbc.Alert(
            f"{filename} loaded successfully, but failed to make JSON serializable: {e}",
            color="danger",
        )

    return data_table, None


def parse_audio_contents(
    filenames: list,
    contents: list,
    feature_extraction_args: dict,
) -> tuple:
    # Check extensions
    bad_files = [
        fn
        for fn in filenames
        if not os.path.splitext(fn)[-1].lower() in ALLOWED_EXTENSIONS_AUDIO
    ]
    if bad_files:
        return None, dbc.Alert(
            f"Found {len(bad_files)} with unsupported extensions.\n"
            + f"Only {ALLOWED_EXTENSIONS_AUDIO} are supported.",
            color="danger",
            dismissable=True,
        )
    # TODO: check file types and return dbc.Alert in case extension not supported

    # TODO: process contents in order to work with librosa (MFCCs) and/or torchaudio (speaker embeddings)

    # TODO: send processed contents and filenames to FeatureExtractor
    pass
