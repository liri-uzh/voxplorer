import os
import unittest
import numpy as np
import torch
from ..lib.speaker_embeddings import SpeakerEmbedder


class TestSpembed(unittest.TestCase):
    def setUp(self) -> None:
        audio_dir = 'tests/data/'
        metadata_vars = {
            'metavars': ['speaker', '-', 'sentence'],
            'separator': '_'
        }
        self.se = SpeakerEmbedder(audio_dir, metadata_vars)

    def test_audio_files_exist(self):
        self.assertTrue(all(os.path.exists(f)
                        for f in self.se.audio_files))

    def test_load_dir(self):
        result_wavs, result_wav_lens = self.se._load_audio_files()
        self.assertIsInstance(result_wavs, torch.Tensor)
        self.assertEqual(result_wavs.shape[0], 5)
        self.assertIsInstance(result_wav_lens, torch.Tensor)
        self.assertEqual(result_wav_lens.shape[0], 5)

    def test_spembed_file(self):
        se = SpeakerEmbedder('tests/data/01_test_001.wav')
        wavs, wav_lens = se._load_audio_files()
        result = se.spembed(wavs=wavs, wav_lens=wav_lens)
        self.assertIsInstance(result, np.ndarray)
        self.assertEqual(result.shape, (1, 192))

    def test_process_files(self):
        result_features, result_metadata_values, result_metadata_labels, result_feature_labels = self.se.process_files()

        # Type checks
        self.assertIsInstance(result_features, np.ndarray)
        self.assertIsInstance(result_metadata_values, np.ndarray)
        self.assertIsInstance(result_metadata_labels, list)
        self.assertIsInstance(result_feature_labels, list)

        # Result has shape (n_files, n_features)
        self.assertEqual(result_features.shape,
                         (len(self.se.audio_files), 192))
        self.assertEqual(result_metadata_values.shape,
                         (len(self.se.audio_files), 3))
        self.assertEqual(len(result_metadata_labels), 3)
        self.assertEqual(len(result_feature_labels), 192)


if __name__ == '__main__':
    unittest.main()
