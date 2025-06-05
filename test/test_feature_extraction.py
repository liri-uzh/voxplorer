import os
import unittest
import librosa
import torchaudio
from lib.feature_extraction import FeatureExtractor


class TestFeatureExtractor(unittest.TestCase):
    def setUp(self):
        self.audio_path = os.path.join(os.path.dirname(__file__), "data/")
        self.audio_files = [
            os.path.join(os.path.dirname(__file__), "data", f)
            for f in os.listdir(self.audio_path)
        ]
        self.feature_methods = {
            "mel_features": {"delta": True, "summarise": True, "n_mfccs": 10},
            "lpc_features": {"n_lpccs": 5, "summarise": True},
        }
        self.metavars = {"variables": ["speaker", "-"], "split_char": "_"}
        self.audio_librosa, self.sr_librosa = librosa.load(self.audio_files[0], sr=None)
        self.audio_torch, self.sr_torch = torchaudio.load(self.audio_files[0])
        if self.sr_torch != 16000:
            self.audio_torch = torchaudio.functional.resample(
                self.audio_torch, self.sr_torch, 16000
            )
        self.feature_extractor = FeatureExtractor(
            audio_path=self.audio_path,
            feature_methods=self.feature_methods,
            metavars=self.metavars,
        )

    def test_all_exist(self):
        self.assertTrue(all(os.path.exists(f) for f in self.audio_files))

    def test_mel_features(self):
        mel_features = self.feature_extractor.mel_features(
            self.audio_librosa, self.sr_librosa, **self.feature_methods["mel_features"]
        )
        # 10 mfccs, 10 deltas -> summarised -> mean and std per feature
        self.assertEqual(mel_features.shape, (10 * 2 * 2,))

    def test_lpc_features(self):
        lpc_features = self.feature_extractor.lpc_features(
            self.audio_librosa, self.sr_librosa, **self.feature_methods["lpc_features"]
        )
        # 5 lpccs -> summarised -> mean and std per feature
        self.assertEqual(lpc_features.shape, (5 * 2,))

    def test_speaker_embeddings(self):
        speaker_embeddings = self.feature_extractor.speaker_embeddings(self.audio_torch)
        self.assertEqual(speaker_embeddings.shape, (1, 192))

    def test_extract_metadata(self):
        expected = {
            "filename": [os.path.basename(self.audio_files[0])],
            "speaker": [os.path.basename(self.audio_files[0]).split("_")[0]],
        }
        metadata = self.feature_extractor.extract_metadata(
            self.audio_files[0], **self.metavars
        )
        assert dict(metadata) == expected

    def test_process_files(self):
        expected_metadata = {
            "filename": [os.path.basename(f) for f in self.audio_files],
            "speaker": [os.path.basename(f).split("_")[0] for f in self.audio_files],
        }
        expected_feature_labels = (
            [f"mfcc{i + 1}_mean" for i in range(10)]
            + [f"delta{i + 1}_mean" for i in range(10)]
            + [f"mfcc{i + 1}_std" for i in range(10)]
            + [f"delta{i + 1}_std" for i in range(10)]
            + [f"lpcc{i + 1}_mean" for i in range(5)]
            + [f"lpcc{i + 1}_std" for i in range(5)]
        )
        features, metadata = self.feature_extractor.process_files()

        assert dict(metadata) == expected_metadata
        self.assertEqual(features["mfcc1_mean"].shape[0], 2)
        assert list(features.keys()) == expected_feature_labels


if __name__ == "__main__":
    unittest.main()
