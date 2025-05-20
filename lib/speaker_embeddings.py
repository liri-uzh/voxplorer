# Speaker embeddings using ECAPA-TDNN VoxCeleb
# TODO: add speaker verification option

from collections import defaultdict
import warnings
import os
from typing import Tuple
import numpy as np
from speechbrain.pretrained import EncoderClassifier
# from speechbrain.inference.classifiers import EncoderClassifier
import torch
import torch.nn.functional as F
import torchaudio


class SpeakerEmbedder:
    """
    Speaker embedding using ECAPA-TDNN VoxCeleb.

    Parameters:
    -----------
    audio_dir: str
        Directory containing audio files. Can also be a single audio file.
    metadata_vars : dict
        Dictionary of metadata variables to extract from the filename. Keys are 
        metavars: list of names of metadata variables to extract from the filename
        (use '-' to ignore the variable). Should have the same index as variable in 
        filename after splitting by separator;
        separator: the separating character between metadata variables in the filename.
        add_selection_column: whether to add a selection column to the metadata. If True,
        adds a column 'selection' with value 0 for all files.
        Example:
        metadata_vars = {
            'metavars': ['speaker', 'emotion', '-', 'selection'],
            'separator': '_',
            'add_selection_column': False
        }

    Attributes:
    -----------
    audio_dir: str
        Directory containing audio files. Can also be a single audio file.
    metadata_vars : dict
        Dictionary of metadata variables to extract from the filename. Keys are 
        metavars: list of names of metadata variables to extract from the filename
        (use '-' to ignore the variable). Should have the same index as variable in 
        filename after splitting by separator;
        separator: the separating character between metadata variables in the filename.
        Example:
        metadata_vars = {
            'metavars': ['speaker', 'emotion', '-', 'selection'],
            'separator': '_'
        }
    """

    def __init__(
            self, audio_dir: str,
            metadata_vars: dict = {
                'metavars': None, 'separator': None, 'add_selection_column': False}
    ):
        self.audio_dir: str = audio_dir
        self.metadata_vars: dict = metadata_vars
        if os.path.isdir(self.audio_dir):
            self.audio_files: list = [os.path.join(self.audio_dir, f
                                                   ) for f in os.listdir(self.audio_dir) if f.endswith('.wav') and not f.startswith('.')
                                      ]
            if len(self.audio_files) == 0:
                raise FileNotFoundError('No .wav files found in audio_dir')
        elif self.audio_dir.endswith('.wav'):
            self.audio_files: list = [self.audio_dir]
        else:
            raise FileNotFoundError(
                'audio_dir must be a directory containing .wav files or a .wav file')

        # Init pretrained model
        if not os.path.isdir('.pretrained_classifier'):
            os.mkdir('.pretrained_classifier')
        self.classifier: EncoderClassifier = EncoderClassifier.from_hparams(
            source='speechbrain/spkrec-ecapa-voxceleb',
            savedir='.pretrained_classifier'
        )

    # Helper to load audio files as tensors

    def _load_audio_files(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Audio files loader. Loads into a torch.Tensor the wav files making sure 
        that the sampling rate is 16000 Hz. The files are padded with zeros to
        match the length of the longest file. The tensor is then reshaped to
        (N files, max_len) where max_len is the length of the longest file.
        It also creates a torch.Tensor containing the length of each file relative
        to the longest file.

        Returns:
        --------
        wavs: torch.Tensor
            Tensor of audio files.
        wav_lens: torch.Tensor
            Tensor of relative audio file lengths.
        """
        # load files
        wavs: list = [None] * len(self.audio_files)
        max_len = 0
        for i, wav_file in enumerate(self.audio_files):
            signal, sr = torchaudio.load(wav_file)
            if sr != 16000:  # make sure sampling rate = 16000 Hz
                signal = torchaudio.functional.resample(
                    signal, orig_freq=sr, new_freq=16000
                )
            if signal.shape[1] > max_len:
                max_len = signal.shape[1]

            wavs[i] = signal

        # pad files
        wav_lens = [sig.shape[1] / max_len for sig in wavs]
        wavs = [F.pad(sig, (0, max_len - sig.shape[1]),
                      mode='constant', value=0) for sig in wavs]

        # cat tensors
        wavs = torch.cat(wavs, dim=0)  # tensor of audio files
        # tensor of relative audio file lengths
        wav_lens = torch.tensor(wav_lens)

        return wavs, wav_lens

    def spembed(self, wavs, wav_lens) -> np.ndarray:
        """
        Speaker embedding using ECAPA-TDNN VoxCeleb.

        Parameters:
        -----------
        wavs: torch.Tensor
            Tensor of audio files.
        wav_lens: torch.Tensor
            Tensor of relative audio file lengths.

        Returns:
        --------
        np.ndarray
            Speaker embedding (N files, 192).
        """
        # embed
        embedding: torch.Tensor = self.classifier.encode_batch(
            wavs=wavs, wav_lens=wav_lens)

        # reshape and transform to numpy
        embedding: np.ndarray = torch.reshape(embedding, shape=(
            embedding.shape[0], 192)).numpy()  # 192 embeddings

        return embedding

    def extract_metadata(self, metavars: list = None, separator: str = '_', add_selection_column: bool = False) -> dict:
        """
        Extracts metadata variables from the filename.

        Parameters:
        -----------
        metavars: list
            List of metadata variable names. Metadata variable should have the same index as
            variable in filename after splitting by separator. If '-' then variable is skipped.
            If None, returns only filename.
        separator: str
            Separator character between metadata variables.
        add_selection_column: bool
            Whether to add a selection column to the metadata. If True, adds a column 'selection'
            with value 0 for all files.

        Returns:
        --------
        dict
            Dictionary of metadata variables. If metavars is None, returns only filename.
        """
        metadata_dict: defaultdict = defaultdict(list)  # init empty dict

        for f in self.audio_files:
            basename = os.path.basename(f)
            if metavars:
                metadata: list = os.path.splitext(basename)[0].split(separator)

            metadata_dict['filename'].append(basename)
            for i, var in enumerate(metavars):
                if var != '-':
                    metadata_dict[var].append(metadata[i])

            if add_selection_column:
                metadata_dict['selection'].append(0)

        return metadata_dict

    def extract_feature_labels(self, num_features: int = 192) -> list:
        """
        Extracts feature labels.

        Parameters:
        -----------
        num_features: int
            Number of features.

        Returns:
        --------
        list
            List of feature labels.
        """
        return [f'X{i}' for i in range(num_features)]

    def process_files(self):
        """
        Process files.

        Returns:
        --------
        features: np.ndarray
            Speaker embeddings. Shape (N files, 192).
        metadata_values: np.ndarray
            Extracted metadata. Shape is (N files, n_metavars).
        metadata_labels: list
            Metadata variable names.
        feature_labels: list
            Feature variable names.
        """
        # load audio files
        wavs, wav_lens = self._load_audio_files()

        # embed
        features: np.ndarray = self.spembed(wavs=wavs, wav_lens=wav_lens)
        feature_labels: list = self.extract_feature_labels(
            num_features=features.shape[1])

        # extract metadata
        metadata_dict: dict = self.extract_metadata(**self.metadata_vars)

        # convert to numpy
        metadata_labels: list = list(metadata_dict.keys())
        metadata_values: np.ndarray = np.array(list(metadata_dict.values())).T

        return features, metadata_values, metadata_labels, feature_labels


# debugging
if __name__ == '__main__':
    audio_dir = 'tests/data/'
    feature_methods = {
        'mel_features': {'deltas': True, 'summarise': True, 'n_mfcc': 13},
        'acoustic_features': {'f0min': 75.0, 'f0max': 600.0},
        'low_lvl_features': {'use_mean_contrasts': True, 'summarise': True},
        'lpc_features': {'summarise': True}
    }
    metadata_vars = {
        'metavars': ['speaker', '-', 'sentence'],
        'separator': '_'
    }
    se = SpeakerEmbedder(audio_dir, metadata_vars)
    features, metadata_values, metadata_labels = se.process_files()
    print(features.shape)
    print(metadata_values.shape)
    print(metadata_labels)
