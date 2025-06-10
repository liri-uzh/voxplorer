import base64
import io
import os
import unittest

import librosa
import numpy as np
import torchaudio

from lib.feature_extraction import FeatureExtractor


class TestFeatureExtractor(unittest.TestCase):
    def setUp(self):
        audio_path = os.path.join(os.path.dirname(__file__), "data/")
        self.filenames = [
            os.path.join(audio_path, "sp01_sample1.wav"),
            os.path.join(audio_path, "sp02_sample2.wav"),
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
        self.filebytes = []
        for fn in self.filenames:
            with open(fn, "rb") as fb:
                data = fb.read()
                self.filebytes.append(base64.b64encode(data).decode("ascii"))
        self.decoded_files = []
        for fb in self.filebytes:
            header, b64 = fb.split(",", 1)
            audio_bytes = base64.b64decode(b64)
            self.decoded_files.append(io.BytesIO(audio_bytes))
        self.sig_librosa, self.sr_librosa = librosa.load(
            os.path.join(
                os.path.dirname(__file__),
                "data",
                self.filenames[0],
            ),
            sr=None,
        )
        self.sig_torch, self.sr_torch = torchaudio.load(
            os.path.join(
                os.path.dirname(__file__),
                "data",
                self.filenames[0],
            )
        )

    def test_load_audio(self):
        # Test audio load for MFCCs/LPCCs
        y, sr = FeatureExtractor._load_audio(
            audio_file=self.decoded_files[0],
            method="mel_features",
        )

        # sign
        self.assertCountEqual(
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
            audio_file=self.decoded_files[0],
            method="speaker_embeddings",
        )

        # sign
        self.assertCountEqual(
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
        wl = int(0.025 * self.sr_librosa)
        overlap = int(0.010 * self.sr_librosa)
        step = int(wl - overlap)
        n_frames = int((self.sig_librosa.shape[0] / step) + 1)

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
        x = FeatureExtractor.speaker_embeddings(
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
            (1, 192),
            f"x has shape {x.shape} instead of (1, 192).",
        )

    def test_extract_metadata(self):
        expected = {
            "filename": ["sp01_sample1.wav"],
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
        self.assertCountEqual(
            feature_labels,
            expected,
        )

    def test_FeatureExtractor(self):
        # Test init is correct
        fe = FeatureExtractor(
            filenames=self.filenames,
            filebytes=self.filebytes,
            feature_methods=self.feature_methods,
            metavars=self.metavars,
        )

        self.assertIsInstance(
            fe.filenames,
            list,
        )
        self.assertIsInstance(
            fe.decoded_files,
            list,
        )

        self.assertCountEqual(
            fe.filenames,
            self.filenames,
            "fe.filenames is different",
        )
        self.assertCountEqual(
            fe.decoded_files, self.decoded_files, "fe.decoded_files is different"
        )

        # Test exceptions
        with self.assertRaises(ValueError):
            fe = FeatureExtractor(
                filenames=["only1"],
                filebytes=self.filebytes,
                feature_methods=self.feature_methods,
                metavars=self.metavars,
            )
        with self.assertRaises(ValueError):
            fe = FeatureExtractor(
                filenames=["error.ogg", "error.txt"],
                filebytes=self.filebytes,
                feature_methods=self.feature_methods,
                metavars=self.metavars,
            )

    def test_process_files(self):
        fe = FeatureExtractor(
            filenames=self.filenames,
            filebytes=self.filebytes,
            feature_methods=self.feature_methods,
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
        }
        combined_metadata = list(
            zip(
                [
                    "sp01_sample1.wav",
                    "sp02_sample2.wav",
                ],
                [
                    "sp01",
                    "sp02",
                ],
                [
                    "sample1",
                    "sample2",
                ],
            )
        )
        combined_metadata.sort(key=lambda x: x[0])
        sorted_filenames, sorted_speakers, sorted_samples = zip(*combined_metadata)
        expected["metadata"] = {
            "filename": list(sorted_filenames),
            "speaker": list(sorted_speakers),
            "sample": list(sorted_samples),
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
        self.assertCountEqual(
            list(features.keys()),
            expected["feature_labels"],
        )
        self.assertEqual(
            len(features["c1_mean"]),
            expected["n_obs"],
        )

        # Check metadata
        self.assertIsInstance(
            metadata,
            dict,
        )
        combined_metadata = list(
            zip(
                metadata["filename"],
                metadata["speaker"],
                metadata["sample"],
            )
        )
        combined_metadata.sort(key=lambda x: x[0])
        sorted_filenames, sorted_speakers, sorted_samples = zip(*combined_metadata)
        metadata_sorted = {
            "filename": list(sorted_filenames),
            "speaker": list(sorted_speakers),
            "sample": list(sorted_samples),
        }
        self.assertDictEqual(
            metadata_sorted,
            expected["metadata"],
        )
        # Check for elements matching in position
        for i, f in enumerate(metadata["filename"]):
            self.assertIn(
                metadata["speaker"][i],
                f,
                f"Speaker {metadata['speaker'][i]} does not match file {f} at index {
                    i
                }",
            )
            self.assertIn(
                metadata["sample"][i],
                f,
                f"Sample {metadata['sample'][i]} does not match file {f} at index {i}",
            )


if __name__ == "__main__":
    unittest.main()
