# FILEPATH: voxplorer/tests/test_data_manipulator.py

import os
import unittest
import numpy as np
import pandas as pd
from ..lib.data_manager import load_data, split_data, filter_selection


class TestDataManager(unittest.TestCase):
    def setUp(self) -> None:
        self.test_data_path = 'tests/data/data_load_test.csv'
        self.metavars = ['y0', 'y1', 'z']
        self.df = pd.DataFrame({
            'var1': ['var1_1', 'var1_2', 'var1_3'],
            'var2': ['var2_1', 'var2_2', 'var2_3'],
            'feature1': [1, 2, 3],
            'feature2': [4, 5, 6],
        })
        self.df_metavars = ['var1', 'var2']
        self.df_features = ['feature1', 'feature2']
        self.output_file = 'tests/output/data_out_test.csv'

    def test_load_data(self):
        result = load_data(path_to_data=self.test_data_path,
                           metavars=self.metavars)
        self.assertIsInstance(result, pd.DataFrame)
        self.assertEqual(result.shape, (5, 8))

    def test_split_data(self):
        result_X, result_Y, result_metavars = split_data(
            df=self.df, features=self.df_features, metavars=self.df_metavars)
        self.assertIsInstance(result_X, np.ndarray)
        self.assertIsInstance(result_Y, np.ndarray)
        self.assertIsInstance(result_metavars, list)

        self.assertEqual(result_X.shape, (3, 2))
        self.assertEqual(result_Y.shape, (3, 2))
        self.assertEqual(result_metavars, self.df_metavars)

    def test_filter_selection(self):
        filter_selection(df=self.df, output_file=self.output_file,
                         metavar='var1', metavar_filter='var1_1')
        self.assertTrue(os.path.exists(self.output_file))
        self.assertTrue(os.path.exists(
            self.output_file.replace('.csv', '_dtypes.json')))

        result = pd.read_csv(self.output_file)
        self.assertTrue((result['var1'] == 'var1_1').all())
        self.assertEqual(result.shape, (1, 4))

    def tearDown(self) -> None:
        if os.path.exists(self.output_file):
            os.remove(self.output_file)
            os.remove(self.output_file.replace('.csv', '_dtypes.json'))


if __name__ == '__main__':
    unittest.main()
