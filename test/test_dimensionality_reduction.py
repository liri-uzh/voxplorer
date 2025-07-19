"""Unit tests for the dimensionality_reduction module."""
import unittest
import numpy as np
from ..lib.dimensionality_reduction import pca_reduction, umap_reduction, tsne_reduction, mds_reduction


class TestDimensionalityReduction(unittest.TestCase):
    def setUp(self):
        # Create sample data
        self.X = np.random.rand(100, 10)  # 100 samples, 10 features
        # binary labels for supervised method
        self.y = np.random.randint(0, 2, 100)

    def test_pca_reduction(self):
        X_reduced, explained_variance, singular_values = pca_reduction(
            X=self.X, n_components=2)
        self.assertEqual(X_reduced.shape, (100, 2))
        self.assertEqual(explained_variance.shape[0], 2)
        self.assertEqual(singular_values.shape[0], 2)

    def test_umap_reduction(self):
        X_reduced = umap_reduction(X=self.X)
        self.assertEqual(X_reduced.shape, (100, 2))

        # Test supervised method
        X_reduced = umap_reduction(X=self.X, y=self.y)
        self.assertEqual(X_reduced.shape, (100, 2))

    def test_tsne_reduction(self):
        X_reduced = tsne_reduction(X=self.X)
        self.assertEqual(X_reduced.shape, (100, 2))

    def test_mds_reduction(self):
        X_reduced = mds_reduction(X=self.X)
        self.assertEqual(X_reduced.shape, (100, 2))


if __name__ == '__main__':
    unittest.main()
