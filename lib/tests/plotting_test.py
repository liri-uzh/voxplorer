"""Unit tests for the plotting module."""

import unittest
import numpy as np
from ..lib.plotting import scatter_2d, scatter_3d, line_plot, bar_plot, box_plot, histogram, heatmap


class TestPlotting(unittest.TestCase):
    def setUp(self):
        self.X_1d = np.random.rand(100)
        self.X_2d = np.random.rand(100, 2)
        self.X_3d = np.random.rand(100, 3)
        self.y = np.random.randint(0, 2, 100)
        self.z = np.random.rand(100)
        self.title = "Test plot"

    def test_scatter_2d(self):
        fig = scatter_2d(self.X_2d, self.y, title=self.title)
        self.assertIsNotNone(fig)

    def test_scatter_3d(self):
        fig = scatter_3d(self.X_3d, self.y, title=self.title)
        self.assertIsNotNone(fig)

    def test_line_plot(self):
        fig = line_plot(self.X_2d, self.y, title=self.title)
        self.assertIsNotNone(fig)

    def test_bar_plot(self):
        fig = bar_plot(self.X_2d, self.y, title=self.title)
        self.assertIsNotNone(fig)

    def test_box_plot(self):
        fig = box_plot(self.X_2d, self.y, title=self.title)
        self.assertIsNotNone(fig)

    def test_histogram(self):
        fig = histogram(self.X_1d, title=self.title)
        self.assertIsNotNone(fig)

    def test_heatmap(self):
        fig = heatmap(self.X_2d, z=self.z, title=self.title)
        self.assertIsNotNone(fig)


if __name__ == "__main__":
    unittest.main()
