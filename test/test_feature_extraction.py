import os
import unittest
import librosa
import numpy as np
import torchaudio
from lib.feature_extraction import FeatureExtractor


class TestFeatureExtractor(unittest.TestCase):
    def setUp(self):
        self.audio_path = os.path.join(os.path.dirname(__file__), "data/")
        self.audio_files = [
            os.path.join(self.audio_path, "sp01_sample1.wav"),
            os.path.join(self.audio_path, "sp02_sample1.wav"),
        ]
        self.metavars = {
            "variables": ["speaker", "sample"],
            "split_char": "_",
        }
        self.feature_methods = {
            "mel_features": {
                "n_mfccs": 10,
                "n_mels": 40,
                "delta": True,
                "summarise": True,
            }
        }
        self.sig_librosa, self.sr_librosa = librosa.load(
            self.audio_files[0],
            sr=None,
        )
        self.sig_torch, self.sr_torch = torchaudio.load(self.audio_files[0])

    def test_load_audio(self):
        # Test audio load for MFCCs/LPCCs
        y, sr = FeatureExtractor._load_audio(
            audio_file=self.audio_files[0],
            method="mel_features",
        )

        # sign
        self.assertListEqual(
            y.tolist(),
            self.sig_librosa.tolist(),
            "Librosa signals don't match.",
        )
        # sampling rate
        self.assertEqual(
            sr,
            self.sr_librosa,
            "Librosa sampling rates don't match.",
        )

        # Test audio load for torch
        y, sr = FeatureExtractor._load_audio(
            audio_file=self.audio_files[0],
            method="speaker_embeddings",
        )

        # sign
        self.assertListEqual(
            y.tolist(),
            self.sig_torch.tolist(),
            "Torch signals don't match.",
        )
        # sampling rate
        self.assertEqual(
            sr,
            16000,
            "Torch sampling rates don't match.",
        )

    def test_summarise(self):
        X = np.vstack(
            (
                np.zeros((5, 10)),
                np.ones((5, 10)),
            )
        )
        summarised = FeatureExtractor._summarise(features=X)

        # Assert an array is returned
        self.assertIsInstance(
            summarised,
            np.ndarray,
            f"Summarised array is {type(summarised)} instead of a numpy array.",
        )

        # Assert shape
        self.assertTupleEqual(
            summarised.shape,
            (20,),
            f"Summarised array has shape {summarised.shape} instead of (20,)",
        )

        # Assert correct values
        self.assertEqual(
            np.sum(summarised),
            0.5 * 20,
            f"Summarised array sums to {np.sum(summarised)} instaed of 10.0",
        )

    def test_mel_features_no_summary(self):
        # no summary
        x = FeatureExtractor.mel_features(
            audio=self.sig_librosa,
            sr=self.sr_librosa,
            n_mfccs=13,
            n_mels=40,
            win_length=25.0,
            overlap=10.0,
            fmin=100.0,
            fmax=6000.0,
            preemphasis=0.95,
            lifter=22.0,
            delta=True,
            delta_delta=True,
            summarise=False,
        )

        # prep expected values
        # win_length - overlap (in samples)
        wl = 0.025 * self.sr_librosa
        step = (0.025 - 0.01) * self.sr_librosa
        n_frames = ((self.sig_librosa.shape - wl) / step) + 1
        n_frames = int(n_frames)

        # Assert correct type
        self.assertIsInstance(
            x,
            np.ndarray,
            f"x is {type(x)} instead of a numpy array.",
        )

        # Assert correct number of coefficients and frames
        self.assertTupleEqual(
            x.shape,
            (n_frames, 13 * 3),
            f"x has shape {x.shape} instead of {(n_frames, 13 * 3)}",
        )

    def test_mel_features_with_summary(self):
        # with summary
        x_summ = FeatureExtractor.mel_features(
            audio=self.sig_librosa,
            sr=self.sr_librosa,
            n_mfccs=13,
            n_mels=40,
            win_length=25.0,
            overlap=10.0,
            fmin=100.0,
            fmax=6000.0,
            preemphasis=0.95,
            lifter=22.0,
            delta=True,
            delta_delta=True,
            summarise=True,
        )

        # Assert correct type
        self.assertIsInstance(
            x_summ,
            np.ndarray,
            f"x_summ is {type(x_summ)} instead of a numpy array.",
        )

        # Assert correct number of coefficients and frames
        self.assertTupleEqual(
            x_summ.shape,
            (13 * 3 * 2,),
            f"x_summ has shape {x_summ.shape} instead of {(13 * 3 * 2,)}",
        )

    def test_speaker_embeddings(self):
        x = FeatureExtractor(
            audio=self.sig_torch,
            model="speechbrain/spkrec-ecapa-voxceleb",
        )

        # Assert correct type
        self.assertIsInstance(
            x,
            np.ndarray,
            f"x is {type(x)} instead of numpy array.",
        )

        # Assert correct dimensions
        self.assertTupleEqual(
            x.shape,
            (1, 392),
            f"x has shape {x.shape} instead of (1, 392).",
        )

    def test_extract_metadata(self):
        expected = {
            "speaker": ["sp01"],
            "sample": ["sample1"],
        }
        metadata = FeatureExtractor.extract_metadata(
            filename=self.audio_files[0],
            **self.metavars,
        )

        self.assertDictEqual(
            metadata,
            expected,
        )

    def test_add_feature_labels(self):
        feature_labels = FeatureExtractor.add_feature_labels(self.feature_methods)

        # Assert correct type
        self.assertIsInstance(
            feature_labels,
            list,
        )

        expected = (
            [f"c{i + 1}_mean" for i in range(10)]
            + [f"d{i + 1}_mean" for i in range(10)]
            + [f"c{i + 1}_std" for i in range(10)]
            + [f"d{i + 1}_std" for i in range(10)]
        )
        # Assert correct values
        self.assertListEqual(
            feature_labels,
            expected,
        )

    def test_FeatureExtractor(self):
        # Test init is correct
        fe = FeatureExtractor(
            audio_path=self.audio_files[0],
            feature_methods=self.feature_methods,
            metavars=self.metavars,
        )

        self.assertIsInstance(
            fe.audio_files,
            list,
        )
        self.assertListEqual(
            fe.audio_files,
            list(self.audio_files[0]),
            "fe.audio_files is different when initiated with a file",
        )

        fe = FeatureExtractor(
            audio_path=self.audio_path,
            feature_methods=self.feature_methods,
            metavars=self.metavars,
        )
        self.assertIsInstance(
            fe.audio_files,
            list,
        )
        self.assertListEqual(
            fe.audio_files,
            self.audio_files,
            "fe.audio_files is different when initiated with a directory",
        )

    def test_process_files(self):
        fe = FeatureExtractor(
            audio_path=self.audio_path,
            feature_mthods=self.feature_methods,
            metavars=self.metavars,
        )

        features, metadata = fe.process_files()

        expected = {
            "feature_num": 40,
            "feature_labels": [f"c{i + 1}_mean" for i in range(10)]
            + [f"d{i + 1}_mean" for i in range(10)]
            + [f"c{i + 1}_std" for i in range(10)]
            + [f"d{i + 1}_std" for i in range(10)],
            "n_obs": 2,
            "metadata": {
                "speaker": ["sp01", "sp02"],
                "sample": ["sample1", "sample1"],
            },
        }

        # Check features
        self.assertIsInstance(
            features,
            dict,
        )
        self.assertEqual(
            len(features.keys()),
            expected["feature_num"],
        )
        self.assertListEqual(
            list(features.keys()),
            expected["feature_labels"],
        )
        self.assertEqual(
            len(features["c1_mean"]),
            expexted["n_obs"],
        )

        # Check metadata
        self.assertIsInstance(
            metadata,
            dict,
        )
        self.assertDictEqual(
            metadata,
            expected["metadata"],
        )


if __name__ == "__main__":
    unittest.main()
