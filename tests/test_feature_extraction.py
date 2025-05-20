import numpy as np
import os
import unittest
import librosa
import parselmouth
from ..lib.feature_extraction import FeatureExtractor


class TestFeatureExtractor(unittest.TestCase):
    def setUp(self) -> None:
        audio_dir = 'tests/data/'
        audio_file = 'tests/data/01_test_001.wav'
        feature_methods = {
            'mel_features': {'deltas': True, 'summarise': True, 'n_mfcc': 13},
            'acoustic_features': {'f0min': 75.0, 'f0max': 600.0},
            'low_lvl_features': {'use_mean_contrasts': True, 'summarise': True},
            'lpc_features': {'n_lpcc': 13, 'summarise': True}
        }
        metadata_vars = {
            'metavars': ['speaker', '-', 'sentence'],
            'separator': '_'
        }
        self.fe = FeatureExtractor(audio_dir, feature_methods, metadata_vars)
        self.y, self.sr = librosa.load(audio_file, sr=None)
        self.sound = parselmouth.Sound(audio_file)

    def test_audio_files_exist(self):
        self.assertTrue(all(os.path.exists(f)
                        for f in self.fe.audio_files))

    def test_mel_features(self):
        result = self.fe.mel_features(y=self.y, sr=self.sr)
        self.assertEqual(result.shape[1], 13)

        result = self.fe.mel_features(y=self.y, sr=self.sr, deltas=True)
        self.assertEqual(result.shape[1], 13*3)

        result = self.fe.mel_features(y=self.y, sr=self.sr, summarise=True)
        self.assertEqual(result.shape, (13*2,))

        result = self.fe.mel_features(
            y=self.y, sr=self.sr, summarise=True, deltas=True)
        self.assertEqual(result.shape, (13*6,))

    def test_acoustic_features(self):
        result = self.fe.acoustic_features(sound=self.sound)

        # Result is numpy array with shape (18,)?
        self.assertIsInstance(result, np.ndarray)
        self.assertEqual(result.shape, (18,))

        # Array of floats?
        for feature in result:
            self.assertIsInstance(feature, float)

    def test_low_lvl_features(self):
        result = self.fe.low_lvl_features(y=self.y, sr=self.sr)

        # Result is numpy array of 12 features?
        self.assertIsInstance(result, np.ndarray)
        self.assertEqual(result.shape[1], 12)

        # Test with use_mean_contrasts=True
        result = self.fe.low_lvl_features(
            y=self.y, sr=self.sr, use_mean_contrasts=True)

        # Result is numpy array of 6 features?
        self.assertIsInstance(result, np.ndarray)
        self.assertEqual(result.shape[1], 6)

        # Test with summarise = True
        result = self.fe.low_lvl_features(y=self.y, sr=self.sr, summarise=True)

        # Result is numpy array of 24 features?
        self.assertIsInstance(result, np.ndarray)
        self.assertEqual(result.shape, (24,))

        # Test with summarise = True and use_mean_contrasts=True
        result = self.fe.low_lvl_features(
            y=self.y, sr=self.sr, summarise=True, use_mean_contrasts=True)

        # Result is numpy array of 12 features?
        self.assertIsInstance(result, np.ndarray)
        self.assertEqual(result.shape, (12,))

    def test_lpc_features(self):
        result = self.fe.lpc_features(y=self.y, sr=self.sr)

        # Result is numpy array with shape (13,)?
        self.assertIsInstance(result, np.ndarray)
        self.assertEqual(result.shape[1], 13)

        # Test with different n_lpcc
        result = self.fe.lpc_features(y=self.y, sr=self.sr, n_lpcc=20)
        self.assertIsInstance(result, np.ndarray)
        self.assertEqual(result.shape[1], 20)

        # Test with summarise=True
        result = self.fe.lpc_features(y=self.y, sr=self.sr, summarise=True)
        self.assertIsInstance(result, np.ndarray)
        self.assertEqual(result.shape, (13*2,))

    def test_process_files(self):
        result_features, result_metavalues, result_metalabel, result_featlabel = self.fe.process_files()

        # Type checks
        self.assertIsInstance(result_features, np.ndarray)
        self.assertIsInstance(result_metavalues, np.ndarray)
        self.assertIsInstance(result_metalabel, list)
        self.assertIsInstance(result_featlabel, list)

        # Result has shape (n_files, n_features)?
        self.assertEqual(result_features.shape,
                         (len(self.fe.audio_files), 13*6+18+12+13*2))
        self.assertEqual(result_metavalues.shape,
                         (len(self.fe.audio_files), 3))
        self.assertEqual(len(result_metalabel), 3)
        self.assertEqual(len(result_featlabel), 13*6+18+12+13*2)


if __name__ == '__main__':
    unittest.main()
