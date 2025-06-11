import os
import base64
import io
import dash_bootstrap_components as dbc
import polars as pl
from .feature_extraction import FeatureExtractor

ALLOWED_EXTENSIONS_AUDIO = {".wav", ".mp3", ".flac"}


def parse_table_contents(contents, filename):
    contents_type, content_string = contents.split(",")

    # decode contents
    decoded = base64.b64decode(content_string)

    try:
        # generate file stream
        file_stream = io.BytesIO(decoded)

        ext = os.path.splitext(filename)[-1]
        if ext == ".csv":
            data_table = pl.read_csv(file_stream)
        elif ext == ".tsv":
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
                dismissable=True,
            )
    except Exception as e:
        return None, dbc.Alert(
            f"There was an error processing file {filename}: {e}",
            color="danger",
            dismissable=True,
        )

    # Make JSON serialisable
    try:
        data_table = data_table.to_dicts()
    except Exception as e:
        return None, dbc.Alert(
            f"{filename} loaded successfully, but failed to make JSON serializable: {e}",
            color="danger",
            dismissable=True,
        )

    return data_table, None


def parse_audio_contents(
    contents: list,
    filenames: list,
    feature_extraction_args: dict,
    metavars: dict,
) -> tuple:
    try:
        # Initialise feature extractor
        fe = FeatureExtractor(
            filenames=filenames,
            filebytes=contents,
            feature_methods=feature_extraction_args,
            metavars=metavars,
        )
    except Exception as e:
        return None, dbc.Alert(
            e,
            color="danger",
            dismissable=True,
        )

    # Process the files
    try:
        features, metadata = fe.process_files()
    except Exception as e:
        return None, dbc.Alert(
            f"Error extracting features: {e}",
            color="danger",
            dismissable=True,
        )

    # process metavariables and features in table
    try:
        data_table = pl.DataFrame(metadata | features)
    except Exception as e:
        dbc.Alert(
            f"Error while creating features table: {e}",
            color="danger",
            dismissable=True,
        )

    # Make JSON serialisable
    try:
        data_table = data_table.to_dicts()
    except Exception as e:
        return None, dbc.Alert(
            f"Failed to make JSON serializable: {e}",
            color="danger",
            dismissable=True,
        )

    return data_table, None
