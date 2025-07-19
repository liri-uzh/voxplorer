import base64
import io
import unittest
from unittest.mock import patch

import dash_bootstrap_components as dbc
import polars as pl

from lib.data_loader import parse_audio_contents, parse_table_contents


def encode_content(raw_bytes: bytes) -> str:
    """Produce the Dash-upload style 'data:;base64,…' string."""
    b64 = base64.b64encode(raw_bytes).decode("utf-8")
    return f"data:;base64,{b64}"


class TestParseTableContents(unittest.TestCase):
    def test_parse_csv_success(self):
        # prepare a small CSV in memory
        df = pl.DataFrame({"a": [1, 2], "b": ["x", "y"]})
        buf = io.BytesIO()
        df.write_csv(buf)
        contents = encode_content(buf.getvalue())

        data, alert = parse_table_contents(contents, "test.csv")
        self.assertIsNone(alert)
        self.assertIsInstance(data, list)
        self.assertEqual(data, [{"a": 1, "b": "x"}, {"a": 2, "b": "y"}])

    def test_parse_tsv_success(self):
        df = pl.DataFrame({"c": [3, 4], "d": ["m", "n"]})
        buf = io.BytesIO()
        df.write_csv(buf, separator="\t")
        contents = encode_content(buf.getvalue())

        data, alert = parse_table_contents(contents, "test.tsv")
        self.assertIsNone(alert)
        self.assertEqual(data, [{"c": 3, "d": "m"}, {"c": 4, "d": "n"}])

    def test_parse_excel_success(self):
        df = pl.DataFrame({"x": [10], "y": [20]})
        buf = io.BytesIO()
        df.write_excel(buf)
        raw = buf.getvalue()
        for ext in (".xls", ".xlsx", ".xlsb"):
            contents = encode_content(raw)
            data, alert = parse_table_contents(contents, f"sheet{ext}")
            self.assertIsNone(alert)
            self.assertEqual(data, [{"x": 10, "y": 20}])

    def test_unsupported_extension_txt(self):
        contents = encode_content(b"foo,bar")
        data, alert = parse_table_contents(contents, "data.txt")
        self.assertIsNone(data)
        self.assertIsInstance(alert, dbc.Alert)
        self.assertIn("modify the extension", alert.children)

    def test_unsupported_extension_unknown(self):
        contents = encode_content(b"anything")
        data, alert = parse_table_contents(contents, "data.foo")
        self.assertIsNone(data)
        self.assertIsInstance(alert, dbc.Alert)
        self.assertIn("could not be decoded", alert.children)

    @patch("lib.data_loader.pl.read_csv", side_effect=ValueError("boom"))
    def test_decode_error(self, mock_read_csv):
        contents = encode_content(b"1,2,3")
        data, alert = parse_table_contents(contents, "file.csv")
        self.assertIsNone(data)
        self.assertIsInstance(alert, dbc.Alert)
        self.assertIn(
            "There was an error processing file file.csv: boom", alert.children
        )

    def test_json_serialization_fail(self):
        # monkeypatch read_csv to return an object whose to_dicts raises
        class FakeDF:
            def to_dicts(self):
                raise RuntimeError("nope")

        with patch("lib.data_loader.pl.read_csv", return_value=FakeDF()):
            contents = encode_content(b"irrelevant")
            data, alert = parse_table_contents(contents, "file.csv")
            self.assertIsNone(data)
            self.assertIsInstance(alert, dbc.Alert)
            self.assertIn("failed to make JSON serializable", str(alert.children))


class DummyFE:
    def __init__(self, *args, **kwargs):
        pass

    def process_files(self):
        # return (features, metadata)
        return {"f1": [0.1, 0.2]}, {"m1": ["a", "b"]}


class TestParseAudioContents(unittest.TestCase):
    @patch("lib.data_loader.FeatureExtractor", DummyFE)
    def test_success(self):
        filenames = ["one.wav", "two.mp3"]
        contents = [b"bytes1", b"bytes2"]
        feats_args = {"mfcc": 13}
        metavars = {"id": [1, 2]}

        data, alert = parse_audio_contents(filenames, contents, feats_args, metavars)
        self.assertIsNone(alert)
        expected = [
            {"m1": "a", "f1": 0.1},
            {"m1": "b", "f1": 0.2},
        ]
        self.assertEqual(data, expected)

    @patch("lib.data_loader.FeatureExtractor", side_effect=RuntimeError("init fail"))
    def test_extractor_init_failure(self, mock_fe):
        data, alert = parse_audio_contents([], [], {}, {})
        self.assertIsNone(data)
        self.assertIsInstance(alert, dbc.Alert)
        self.assertIn("init fail", str(alert.children))

    def test_process_files_failure(self):
        class FE2:
            def __init__(self, *a, **kw):
                pass

            def process_files(self):
                raise ValueError("pffft")

        with patch("lib.data_loader.FeatureExtractor", FE2):
            data, alert = parse_audio_contents(["a.wav"], [b""], {}, {})
            self.assertIsNone(data)
            self.assertIsInstance(alert, dbc.Alert)
            self.assertIn("Error extracting features: pffft", alert.children)

    def test_table_construction_failure(self):
        class FE3:
            def __init__(self, *a, **kw):
                pass

            def process_files(self):
                return {"f": [1]}, {"m": ["x"]}

        def bad_dataframe(*args, **kwargs):
            raise RuntimeError("no table")

        with (
            patch("lib.data_loader.FeatureExtractor", FE3),
            patch("lib.data_loader.pl.DataFrame", bad_dataframe),
        ):
            # This will either hit the table exception or serialization exception
            data, alert = parse_audio_contents(["a.wav"], [b""], {}, {})
            # We accept either a failure alert or None data
            self.assertTrue(data is None or isinstance(alert, dbc.Alert))

    def test_serialization_failure(self):
        class DF:
            def __init__(self, data):
                pass

            def to_dicts(self):
                raise RuntimeError("cannot json")

        class FE4:
            def __init__(self, *a, **kw):
                pass

            def process_files(self):
                return {"f": [1]}, {"m": ["x"]}

        with (
            patch("lib.data_loader.FeatureExtractor", FE4),
            patch("lib.data_loader.pl.DataFrame", DF),
        ):
            data, alert = parse_audio_contents(["a.wav"], [b""], {}, {})
            self.assertIsNone(data)
            self.assertIsInstance(alert, dbc.Alert)
            self.assertIn("Failed to make JSON serializable", alert.children)


if __name__ == "__main__":
    unittest.main()
