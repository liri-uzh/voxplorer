"""Feature extraction module.

Classes
-------
FeatureExtractor
    Class to extract acoustic features from audio files.
"""

import base64
import io
import os
import warnings
from collections import defaultdict
from typing import Tuple

import librosa
import numpy as np
import torch
import torchaudio
from speechbrain.inference.classifiers import EncoderClassifier


class FeatureExtractor:
    """Class to extract acoustic features from audio files.

    Parameters
    ----------
    filenames : list[str]
        Filenames of audio files.
    filebytes : list[str]
        Contents of audio files. E.g. 'data:audio/wav;base64,XXXXXXX...'.
    feature_methods: dict
        Dictionary containing feature methods. Keys are the method names,
        values are the arguments passed to the method. Accepted methods are
        'mel_features', 'lpc_features', and 'speaker_embeddings'.
        Example:
            feature_methods = {
                'mel_features': {'n_mfccs' : 13, 'n_mels': 40, 'win_length': 25.0,
                                'overlap': 10.0, 'delta': True, 'delta_delta': False,
                                'summarise': True},
                'speaker_embeddings': {'model': 'speechbrain/spkrec-ecapa-voxceleb'}
            }
    metavars: dict, optional
        Dictionary containing metadata variables to extract. They should have the same
        index as the variable in the filename split by `split_char`. If '-' then that variable is ignored.
        If `variables` is None, returns only the filename. Default is None.
        Example:
            metavars = {
                'variables': ['speaker', 'emotion', '-'],
                'split_char': '_'
            }

    Attributes
        ----------
        filenames: list[str]
            Audio filenames.
        decoded_files: list[io.BufferedIOBase]
            Decoded file bytes.
        feature_methods: dict
            Dictionary containing feature methods.
        metavars: dict
            Dictionary containing metadata variables to extract.

    Methods
    -------
    mel_features
        Extracts mfccs (and delta and delta-delta) from audio file.
    speaker_embeddings
        Extracts speaker embeddings from audio files.
    extract_metadata
        Extracts metadata features from audio filenames.
    add_feature_labels
        Adds labels to feature methods.
    process_files
        Processes audio files and extracts features. (Main method calling the
        other methods depending on the feature_methods dictionary).
    """

    def __init__(
        self,
        filenames: list,
        filebytes: list,
        feature_methods: dict,
        metavars: dict = {"variables": None, "split_char": "_"},
    ) -> None:
        """Initialises the FeatureExtractor class.

        Parameters
        ----------
        filenames : list[str]
            Filenames of audio files.
        filebytes : list[str]
            Contents of audio files. E.g. 'data:audio/wav;base64,XXXXXXX...'.
        feature_methods : dict
            Dictionary containing feature methods. Keys are the method names,
            values are the arguments passed to the method. Accepted methods are
            'mel_features', and 'speaker_embeddings'.
        metavars : dict, optional
            Dictionary containing metadata variables to extract. They should have the same
            index as the variable in the filename split by `split_char`. If '-'
            then that variable is ignored. If None, returns only the filename.
            Default is None.
        """
        if len(filenames) != len(filebytes):
            raise ValueError(
                "`filenames` and `filebytes` must have the same length;\n"
                + f"got {len(filenames)} and {len(filebytes)}"
            )

        self.feature_methods = feature_methods
        self.metavars = metavars

        # Validate files
        try:
            FeatureExtractor._validate_files(filenames)
            self.filenames = filenames
        except Exception as e:
            raise e

        # Decode filebytes
        try:
            self.decoded_files = FeatureExtractor._decode_filebytes(filebytes)
        except Exception as e:
            raise e

    @staticmethod
    def _validate_files(filenames: list):
        ALLOWED_EXTENSIONS = {".wav", ".mp3", ".flac"}
        bad_files = [
            fn
            for fn in filenames
            if os.path.splitext(fn)[-1].lower() not in ALLOWED_EXTENSIONS
        ]
        if bad_files:
            raise ValueError(
                f"Found {len(bad_files)} with unsupported extensions.\n"
                + f"Only {ALLOWED_EXTENSIONS} are supported."
            )

    @staticmethod
    def _decode_filebytes(filebytes: list):
        decoded_files = []
        for fb in filebytes:
            header, b64 = fb.split(",", 1)
            audio_bytes = base64.b64decode(b64)
            decoded_files.append(io.BytesIO(audio_bytes))

        return decoded_files

    # Helper to summarise features by utterance
    @staticmethod
    def _summarise(features: np.ndarray) -> np.ndarray:
        return np.concatenate(
            (np.mean(features, axis=0), np.std(features, axis=0)), axis=0
        )

    # Helpers for delta features
    @staticmethod
    def _delta(mfccs: np.ndarray) -> np.ndarray:
        return librosa.feature.delta(mfccs)

    @staticmethod
    def _delta_delta(mfccs: np.ndarray) -> np.ndarray:
        return librosa.feature.delta(mfccs, order=2)

    # MFCCs
    @staticmethod
    def mel_features(
        audio: np.ndarray,
        sr: int,
        n_mfccs: int = 13,
        n_mels: int = 40,
        win_length: float = 25.0,
        overlap: float = 10.0,
        fmin: float = 100.0,
        fmax: float = 6000.0,
        preemphasis: float = 0.95,
        lifter: float = 22.0,
        delta: bool = False,
        delta_delta: bool = False,
        summarise: bool = False,
    ) -> np.ndarray:
        """Extracts mfccs (and delta and delta-delta) from audio file.

        Parameters
        ----------
        audio: np.ndarray
            Audio signal.
        sr: int
            Sampling rate.
        n_mfccs: int, optional
            Number of mfccs to extract.
        n_mels: int, optional
            Number of mel bands.
        win_length: float, optional
            Window length in milliseconds.
        overlap: float, optional
            Overlap in milliseconds.
        fmin: float, optional
            Minimum frequency.
        fmax: float, optional
            Maximum frequency.
        preemphasis: float, optional
            Preemphasis coefficient.
        lifter: float, optional
            Liftering coefficient.
        delta: bool, optional
            Include delta features. Default is False.
        delta_delta: bool, optional
            Include delta-delta features. Default is False.
        summarise: bool, optional
            Summarise features by utterance. Default is False.

        Returns
        -------
        np.ndarray
            MFCCs (and delta and delta-delta if specified).
            If summarise is True, returns a single vector per utterance containing
            the mean and standard deviation of each feature.
        """
        # Preemphasis
        y = librosa.effects.preemphasis(audio, coef=preemphasis)

        # Compute frame length and hop length
        n_fft = int(sr * win_length / 1000)
        hop_length = int(sr * (win_length - overlap) / 1000)

        # Raise warning if n_mfccs > n_mels
        if n_mfccs > n_mels:
            warnings.warn(
                f"Number of MFCCs ({n_mfccs}) is greater than number of "
                + f"Mel bands ({n_mels}). Setting n_mfccs to {n_mels}."
            )
            n_mfccs = n_mels

        # Extract mfccs
        features = librosa.feature.mfcc(
            y=y,
            sr=sr,
            n_mfcc=n_mfccs,
            n_mels=n_mels,
            n_fft=n_fft,
            hop_length=hop_length,
            fmin=fmin,
            fmax=fmax,
            lifter=lifter,
        )

        # Extract delta features
        if delta and delta_delta:
            features = np.concatenate(
                (
                    features,
                    FeatureExtractor._delta(features),
                    FeatureExtractor._delta_delta(features),
                ),
                axis=0,
            )
        elif delta:
            features = np.concatenate(
                (
                    features,
                    FeatureExtractor._delta(features),
                ),
                axis=0,
            )
        elif delta_delta:
            features = np.concatenate(
                (features, FeatureExtractor._delta_delta(features)), axis=0
            )

        # Summarise features by utterance
        if summarise:
            features = FeatureExtractor._summarise(
                features.T
            )  # n_frames x n_features*2
        else:
            features = features.T  # n_frames x n_features

        # Return transposed features (n_frames x n_features)
        return features

    # Speaker embeddings
    @staticmethod
    def speaker_embeddings(
        audio: torch.Tensor,
        model: str = "speechbrain/spkrec-ecapa-voxceleb",
    ) -> np.ndarray:
        """Extracts speaker embeddings from audio files.

        Parameters
        ----------
        audio: torch.Tensor
            audio tensor.
        model: str, optional
            Speechbrain specific speaker embedding model. Default is 'speechbrain/spkrec-ecapa-voxceleb'.

        Returns
        -------
        np.ndarray
            Speaker embeddings.
        """
        # Init pre-trained model
        if not os.path.isdir(".pretrained_spkrec_models"):
            os.mkdir(".pretrained_spkrec_models")

        # Load model
        classifier = EncoderClassifier.from_hparams(
            source=model, savedir=".pretrained_spkrec_models"
        )

        # Extract embeddings
        embeddings = classifier.encode_batch(wavs=audio)

        # Reshape and transform to numpy
        embeddings = torch.reshape(
            embeddings, shape=(embeddings.shape[1], embeddings.shape[-1])
        ).numpy()

        return embeddings

    # Extract metadata features from filenames
    @staticmethod
    def extract_metadata(
        filename: str,
        variables: list = None,
        split_char: str = "_",
    ) -> dict:
        """Extracts metadata features from audio filenames.

        Parameters
        ----------
        filename: str
            Audio filename.
        variables: list, optional
            List of metadata variables to extract. They should have the same
            index as the variable in the filename split by `split_char`. If '-'
            then that variable is ignored. If None, returns only the filename.
            Default is None.
        split_char: str, optional
            Character to split the filename. Default is '_'.

        Returns
        -------
        dict
            Dictionary containing metadata features.
        """
        # Create empty dict
        metadata = defaultdict(list)

        # Extract metadata from filename
        basename = os.path.basename(filename)
        # Split filename
        if variables is not None:
            metadata_f = os.path.splitext(basename)[0].split(split_char)

            # Extract metadata
            for i, var in enumerate(variables):
                if var != "-":
                    metadata[var].append(metadata_f[i])

        # Extract filename
        metadata["filename"].append(basename)

        return metadata

    # Add feature labels
    @staticmethod
    def add_feature_labels(feature_methods: dict) -> list:
        """Adds labels to feature methods.

        Parameters
        ----------
        feature_methods: dict
            Dictionary containing feature methods. Keys are the method names,
            values are the arguments passed to the method. Accepted methods are
            'mel_features', 'lpc_features', and 'speaker_embeddings'.

        Returns
        -------
        list
            List of feature methods with labels.
        """
        # Add labels to feature methods
        feature_labels = []
        for method, args in feature_methods.items():
            if method == "mel_features":
                tmp_feature_labels = [f"c{i + 1}" for i in range(args["n_mfccs"])]
                if "delta" in args and args["delta"]:
                    tmp_feature_labels.extend(
                        [f"d{i + 1}" for i in range(args["n_mfccs"])]
                    )
                if "delta_delta" in args and args["delta_delta"]:
                    tmp_feature_labels.extend(
                        [f"dd{i + 1}" for i in range(args["n_mfccs"])]
                    )
                if "summarise" in args and args["summarise"]:
                    feature_labels.extend(
                        [f"{label}_mean" for label in tmp_feature_labels]
                        + [f"{label}_std" for label in tmp_feature_labels]
                    )
                else:
                    feature_labels.extend(tmp_feature_labels)
            elif method == "speaker_embeddings":
                feature_labels.extend([f"X{i + 1}" for i in range(args["n_features"])])
            else:
                raise ValueError("Invalid feature method.")

        return feature_labels

    # Helper for loading audio files
    @staticmethod
    def _load_audio(audio_file: io.BytesIO, method: str) -> Tuple[np.ndarray, int]:
        """Loads audio file and returns audio signal and sampling rate.

        Parameters
        ----------
        audio_file: str
            Audio file decoded bytes.
        method: str
            Feature extraction method. Accepted methods are 'mel_features', and 'speaker_embeddings'.

        Returns
        -------
        Tuple[np.ndarray, int]
            Audio signal and sampling rate.
        """
        if method == "mel_features":
            return librosa.load(audio_file, sr=None)

        if method == "speaker_embeddings":
            sig, sr = torchaudio.load(audio_file)
            if sr != 16000:  # resample to 16 kHz
                sig = torchaudio.functional.resample(sig, orig_freq=sr, new_freq=16000)
            return sig, sr

    # Process sound files
    def process_files(self) -> Tuple[dict, dict]:
        """Processes audio files and extracts features.

        Returns
        -------
        dict
            Features dictionary: keys are the feature labels, values are the features (1D numpy arrays).
            Features of shape (n_files x n_frames, n_features) if summarise is False.
            Features of shape (n_files, n_features) if summarise is True.
        dict
            Metadata dictionary: keys are the metadata labels, values are the metadata values.
            Metadata of shape (n_files x n_frames, n_metadata) if summarise is False.
            Metadata of shape (n_files, n_metadata) if summarise is True.
        """
        features = []
        metadata = defaultdict(list)
        feature_labels = FeatureExtractor.add_feature_labels(
            feature_methods=self.feature_methods
        )

        for fn, fb in zip(self.filenames, self.decoded_files):
            # Extract file metadata
            if "variables" in self.metavars and "split_char" in self.metavars:
                metadata_f = FeatureExtractor.extract_metadata(
                    filename=fn,
                    variables=self.metavars["variables"],
                    split_char=self.metavars["split_char"],
                )
            elif "variables" in self.metavars:
                metadata_f = FeatureExtractor.extract_metadata(
                    filename=fn,
                    variables=self.metavars["variables"],
                )
            elif "split_char" in self.metavars:
                metadata_f = FeatureExtractor.extract_metadata(
                    filename=fn,
                    split_char=self.metavars["split_char"],
                )
            else:
                metadata_f = FeatureExtractor.extract_metadata(filename=fn)

            # Extract features
            features_f = []  # list of features for current file
            for method, args in self.feature_methods.items():
                audio, sr = FeatureExtractor._load_audio(audio_file=fb, method=method)
                if method == "mel_features":
                    features_f.append(
                        FeatureExtractor.mel_features(audio=audio, sr=sr, **args)
                    )
                elif method == "speaker_embeddings":
                    features_f.append(
                        FeatureExtractor.speaker_embeddings(audio=audio, **args)
                    )

            # Concatenate features of current file and append to features list
            features_f = np.concatenate(features_f, axis=0)
            # Reshape if format is incorrect
            if len(features_f.shape) == 1:
                features_f = features_f.reshape(1, -1)
            features.append(features_f)

            # Concatenate metadata of current file and append to metadata dict
            for key, value in metadata_f.items():
                # make sure it matches the number of frames
                metadata[key].extend(value * features_f.shape[0])

        # Concatenate features
        features = np.concatenate(features, axis=0)
        # Create dictionary
        features_dict = dict(zip(feature_labels, features.T))

        return features_dict, dict(metadata)
