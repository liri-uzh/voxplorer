import base64
import io
import dash_bootstrap_components as dbc
import polars as pl


def _parse_table(contents, filename):
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

        if "csv" in filename:
            data_table = pl.read_csv(file_stream)
        elif "tsv" in filename:
            data_table = pl.read_csv(file_stream, separator="\t")
        elif "xls" in filename:
            data_table = pl.read_excel(file_stream)
        else:
            return None, dbc.Alert(
                f"File {filename} could not be decoded.",
                color="danger",
            )
    except Exception as e:
        return None, dbc.Alert(
            f"There was an error processing file {filename}: {e}",
            color="danger",
        )

    return data_table, None


def parse_contents(contents, filename, filetype):
    """
    Parse file contents and return dictionary or a DataFrame depending
    on input type.
    """
    data_table = None

    if filetype == "table":
        data_table, alert = _parse_table(contents, filename)

    # TODO:
    # elif filetype == "audio":
    #     data_table = _parse_audio()

    try:
        # make df JSON serializable
        # data_table = data_table.with_row_index()  # add index
        data_table = data_table.to_dicts()
    except Exception as e:
        if alert is not None and data_table is None:
            return None, alert
        else:
            return None, dbc.Alert(
                f"{filename} loaded successfully, but failed storage with exception: {e}\n"
                + "Check that data is JSON serialisable.",
                color="danger",
            )

    return data_table, None
