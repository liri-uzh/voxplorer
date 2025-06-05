import unittest
import pandas as pd
from pandas.testing import assert_frame_equal
from ..lib.db_loader import load_data, update_column_dtype
import os


class TestDbLoader(unittest.TestCase):
    def setUp(self):
        self.table_path = "tests/test_table.csv"
        with open(self.table_path, "w", encoding="utf-8") as f:
            f.write("id,name,age\n1,John,30\n2,Anna,25\n3,Paul,22\n")

        # Create a DataFrame for testing update_column_dtype
        self.data = pd.DataFrame(
            {
                "id": [1, 2, 3],
                "name": ["John", "Anna", "Paul"],
                "age": [30, 25, 22],
            }
        )

    def tearDown(self):
        # remove the temporary files
        if os.path.exists(self.table_path):
            os.remove(self.table_path)

    def test_load_data_noseparator(self):
        expected_df = self.data
        result_df = load_data(self.table_path)
        assert_frame_equal(result_df, expected_df)

    def test_load_data_separator(self):
        expected_df = self.data
        result_df = load_data(self.table_path, sep=",")
        assert_frame_equal(result_df, expected_df)

    def test_load_data_error(self):
        with self.assertRaises(FileNotFoundError):
            load_data("tests/test_table.txt")

    def test_update_column_dtype(self):
        # Convert age to str
        expected_df = pd.DataFrame(
            {
                "id": [1, 2, 3],
                "name": ["John", "Anna", "Paul"],
                "age": ["30", "25", "22"],
            }
        )  # age converted to str
        result_df = update_column_dtype(self.data, "age", "str")
        assert_frame_equal(result_df, expected_df)


if __name__ == "__main__":
    unittest.main()
