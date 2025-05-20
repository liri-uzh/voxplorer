from collections import defaultdict
from typing import Tuple
import os
import warnings
from tkinter import messagebox
import numpy as np
import librosa
import parselmouth
from scipy import signal


class FeatureExtractor:
    """
    Class to extract acoustic features from audio files.

    Parameters:
    -----------
    audio_dir : str
        Path to directory containing audio files. Can also be a path to a single audio file.
    feature_methods : dict
        Dictionary of feature extraction methods to use. Keys are the names of the
        feature extraction methods, values are the arguments to pass to the methods.
        Methods are: 'mel_features', 'acoustic_features', 'low_lvl_features', 'lpc_features'.
        Example:
        feature_methods = {
            'mel_features': {'n_mfcc': 13, 'n_mels': 40, 'win_length': 25, 'overlap': 10, summarise=True},
            'acoustic_features': {'f0min': 75, 'f0max': 600},
            'low_lvl_features': {'win_length': 25, 'overlap': 10},
            'lpc_features': {'n_lpcc': 13, 'win_length': 25, 'overlap': 10}
        }
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
            'add_selection_column': True
        }

    Attributes:
    -----------
    audio_dir : str
        Path to directory containing audio files. Can also be a path to a single audio file.
    feature_methods : dict
        Dictionary of feature extraction methods to use. Keys are the names of the
        feature extraction methods, values are the arguments to pass to the methods.
        Methods are: 'mel_features', 'acoustic_features', 'low_lvl_features', 'lpc_features'.
        Example:
        feature_methods = {
            'mel_features': {'n_mfcc': 13, 'n_mels': 40, 'win_length': 25, 'overlap': 10, summarise=True},
            'acoustic_features': {'f0min': 75, 'f0max': 600},
            'low_lvl_features': {'win_length': 25, 'overlap': 10},
            'lpc_features': {'n_lpcc': 13, 'win_length': 25, 'overlap': 10}
        }
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
    audio_files : list
        List of audio files in audio_dir.
    features : np.ndarray
        Extracted features. Shape is (n_files, n_features) when summarise=True for 
        mel_features, low_lvl_features, lpc_features. If summarise=False, shape is (n_files x n_frames, n_features).
    metadata_dict : defaultdict
        Dictionary of metadata variables. Keys are the names of the metadata variables,
        values are the values of the metadata variables for each audio file.
    """

    def __init__(
        self, audio_dir: str, feature_methods: dict,
        metadata_vars: dict = {'metavars': None,
                               'separator': None, 'add_selection_column': False}
    ):
        self.audio_dir: str = audio_dir
        self.feature_methods: dict = feature_methods
        self.metadata_vars: dict = metadata_vars
        if os.path.isdir(audio_dir):
            self.audio_files: list = [os.path.join(
                self.audio_dir, f) for f in os.listdir(audio_dir) if f.endswith('.wav') and not f.startswith('.')
            ]
            if len(self.audio_files) == 0:
                raise FileNotFoundError('No .wav files found in audio_dir')

        elif self.audio_dir.endswith('.wav'):
            self.audio_files: list = [self.audio_dir]
        else:
            raise FileNotFoundError(
                "audio_dir must be a directory containing .wav files or a .wav file.")

    # helper to summarise features by utterance

    def _summarise_features(self, features_vec: np.ndarray) -> np.ndarray:
        """
        Summarises features by utterance.

        Parameters:
        -----------
        features_vec : np.ndarray
            features vector for each frame in the audio file. Shape is (n_features, n_frames).

        Returns:
        --------
        np.ndarray
            Mean and standard deviation of features for each utterance. Shape is
            (n_features*2,)
        """
        return np.concatenate(
            (np.mean(features_vec, axis=1), np.std(features_vec, axis=1)), axis=0
        )

    # helper of mel_features() to extract delta and delta-delta coefficients

    def _delta_delta(self, mfccs: np.ndarray) -> np.ndarray:
        """
        Computes delta and delta-delta coefficients from MFCCs.

        Parameters:
        -----------
        mfccs : np.ndarray
            MFCCs for each frame in the audio file.

        Returns:
        --------
        features_vec : np.ndarray
            updated features vector containing MFCCs, deltas, and delta-deltas
            for each frame in the audio file. Shape is (n_frames, n_mfccs*3).
        """
        # compute delta and delta-delta
        delta: np.ndarray = librosa.feature.delta(mfccs, order=1)
        delta_delta: np.ndarray = librosa.feature.delta(mfccs, order=2)

        # concatenate features
        features_vec: np.ndarray = np.concatenate(
            (mfccs, delta, delta_delta), axis=0)

        return features_vec

    # pitch helper

    def _pitch(
        self,
        sound: parselmouth.Sound,
        time_step: float = 0.0,
        f0min: float = 75.0,
        f0max: float = 600.0,
        max_candidates: int = 15,
        silence_threshold: float = 0.03,
        voicing_threshold: float = 0.45,
        octave_cost: float = 0.01,
        octave_jump_cost: float = 0.35,
        voiced_unvoiced_cost: float = 0.14
    ) -> Tuple[float, ...]:
        """
        Extracts pitch features from audio file using cross-correlation method.

        Parameters:
        -----------
        sound : parselmouth.Sound
            Parselmouth sound object.
        time_step : float
            Time step in seconds (default=0.0 (=auto)).
        f0min : float
            Minimum pitch frequency in Hz.
        f0max : float
            Maximum pitch frequency in Hz.
        max_candidates : int
            Maximum number of candidates.
        silence_threshold : float
            Threshold for silence.
        voicing_threshold : float
            Threshold for voicing.
        octave_cost : float
            Cost for octave.
        octave_jump_cost : float
            Cost for octave jump.
        voiced_unvoiced_cost : float
            Cost for unvoiced.

        Returns:
        --------
        pitch_features : Tuple[float, ...]
            Pitch features for each frame in the audio file. Shape is (n_frames,).
            The extracted features are:
            - mean pitch
            - median pitch
            - minimum pitch
            - maximum pitch
            - standard deviation of pitch
        """
        pitch: parselmouth.Pitch = parselmouth.praat.call(
            sound, "To Pitch (cc)",
            time_step, f0min, max_candidates, True, silence_threshold,
            voicing_threshold, octave_cost, octave_jump_cost,
            voiced_unvoiced_cost, f0max
        )
        mean_pitch: float = parselmouth.praat.call(
            pitch, 'Get mean', 0, 0, 'Hertz')
        med_pitch: float = parselmouth.praat.call(
            pitch, 'Get quantile', 0, 0, 0.5, 'Hertz')
        min_pitch: float = parselmouth.praat.call(
            pitch, 'Get minimum', 0, 0, 'Hertz', 'Parabolic')
        max_pitch: float = parselmouth.praat.call(
            pitch, 'Get maximum', 0, 0, 'Hertz', 'Parabolic')
        std_pitch: float = parselmouth.praat.call(
            pitch, 'Get standard deviation', 0, 0, 'Hertz')
        return mean_pitch, med_pitch, min_pitch, max_pitch, std_pitch

    # formants helper

    def _formants(
        self,
        sound: parselmouth.Sound,
        maximum_formant: int = 5000
    ) -> Tuple[float, ...]:
        """
        Extracts formants from audio file using Burg method.

        Parameters:
        -----------
        sound : parselmouth.Sound
            Parselmouth sound object.
        maximum_formant : int
            Maximum formant frequency in Hz.

        Returns:
        --------
        formants_features : Tuple[float, ...]
            Mean F1, F2, F3, F4.
        """
        formants: parselmouth.Formant = sound.to_formant_burg(
            maximum_formant=maximum_formant)
        f1: float = parselmouth.praat.call(
            formants, 'Get mean', 1, 0, 0, 'Hertz')
        f2: float = parselmouth.praat.call(
            formants, 'Get mean', 2, 0, 0, 'Hertz')
        f3: float = parselmouth.praat.call(
            formants, 'Get mean', 3, 0, 0, 'Hertz')
        f4: float = parselmouth.praat.call(
            formants, 'Get mean', 4, 0, 0, 'Hertz')

        return f1, f2, f3, f4

    # Helper for vocal tract estimates; code inspired from Voicelab (github.com/Voice-Lab)

    def _vocal_tract_estimates(
        self,
        formants: Tuple[float, ...],
    ) -> Tuple[float, ...]:
        """
        Extract vocal tract estimates measures from audio file.

        Parameters:
        -----------
        formants : Tuple[float, ...]
            Mean formants in the audio file (F1, F2, F3, F4).

        Returns:
        --------
        formant_dispersion : float
            Formant dispersion.
        avg_formant : float
            Average formant.
        geometric_mean : float
            Geometric mean.
        fitch_vtl : float
            Fitch VTL.
        delta_f : float
            VTL Delta f.
        """
        # Extract formant means
        f1: float = formants[0]
        f2: float = formants[1]
        f3: float = formants[2]
        f4: float = formants[3]

        # Formant dispersion
        formant_dispersion: float = (f4 - f1) / 3

        # Average formant
        avg_formant: float = (f1 + f2 + f3 + f4) / 4

        # Geometric mean
        geometric_mean: float = (f1 * f2 * f3 * f4) ** 0.25

        # Fitch VTL
        fitch_vtl: float = ((1 * (35000 / (4 * f1))) + (3 * (35000 / (4 * f2))) +
                            (5 * (35000 / (4 * f3))) + (7 * (35000 / (4 * f4)))) / 4

        # delta f
        xysum: float = (0.5 * f1) + (1.5 * f2) + (2.5 * f3) + (3.5 * f4)
        xsquaredsum: float = (0.5 ** 2) + (1.5 ** 2) + (2.5 ** 2) + (3.5 ** 2)
        delta_f: float = 35000 / (2*xysum / xsquaredsum)

        return formant_dispersion, avg_formant, geometric_mean, fitch_vtl, delta_f

    # Helper for harmonics-to-noise ratio

    def _hnr(
        self,
        sound: parselmouth.Sound,
        time_step: float = 0.01,
        fmin: float = 75.0,
        silence_threshold: float = 0.1,
        periods_per_window: float = 1.0
    ) -> float:
        """
        Extracts harmonics-to-noise ratio from audio file using cc method.

        Parameters:
        -----------
        sound : parselmouth.Sound
            Parselmouth sound object.
        time_step : float
            Time step in seconds (default=0.01).
        fmin : float
            Minimum pitch frequency in Hz.
        silence_threshold : float
            Threshold for silence.
        periods_per_window : float
            Number of periods per window.

        Returns:
        --------
        hnr : float
            Harmonics-to-noise ratio.
        """
        harmonicity: parselmouth.Harmonicity = parselmouth.praat.call(
            sound, "To Harmonicity (cc)",
            time_step, fmin, silence_threshold, periods_per_window
        )
        hnr: float = parselmouth.praat.call(harmonicity, "Get mean", 0, 0)
        return hnr

    # Helper for jitter and shimmer

    def _jitter_shimmer(
        self,
        sound: parselmouth.Sound,
        fmin: float = 75.0,
        fmax: float = 600.0
    ) -> Tuple[float, float]:
        """
        Extracts jitter and shimmer from audio file using cc method.

        Parameters:
        -----------
        sound : parselmouth.Sound
            Parselmouth sound object.
        fmin : float
            Minimum pitch frequency in Hz.
        fmax : float
            Maximum pitch frequency in Hz.

        Returns:
        --------
        jitter : float
            Jitter.
        shimmer : float
            Shimmer.
        """
        point_process: parselmouth.Data = parselmouth.praat.call(
            sound, "To PointProcess (periodic, cc)",
            fmin, fmax
        )

        jitter: float = parselmouth.praat.call(
            point_process, "Get jitter (local)", 0, 0, 0.0001, 0.02, 1.3
        )
        shimmer: float = parselmouth.praat.call(
            [sound,
                point_process], "Get shimmer (local)", 0, 0, 0.0001, 0.02, 1.3, 1.6
        )
        return jitter, shimmer

    # Helper for rms_energy

    def _rms_energy(
        self,
        sound: parselmouth.Sound
    ) -> float:
        """
        Extracts rms energy from audio file.

        Parameters:
        -----------
        sound : parselmouth.Sound
            Parselmouth sound object.

        Returns:
        --------
        rms_energy : float
            Root mean square energy.
        """
        amplitude_square: np.ndarray = sound.values ** 2
        mean_amplitude_square: float = np.mean(amplitude_square)
        rms_energy: float = np.sqrt(mean_amplitude_square)
        return rms_energy

    # MFCCs extraction

    def mel_features(
        self,
        y: np.ndarray,
        sr: int,
        n_mfcc: int = 13,
        n_mels: int = 40,
        win_length: float = 25.0,
        overlap: float = 10.0,
        fmin: float = 150.0,
        fmax: float = 4000.0,
        premphasis: float = 0.95,
        lifter: float = 22,
        deltas: bool = False,
        summarise: bool = False
    ) -> np.ndarray:
        """
        Extracts mfccs (and optionally delta-delta) from audio file.

        Parameters:
        -----------
        y : np.ndarray
            Audio time series.
        sr : int
            Sampling rate of y.
        n_mfcc : int
            Number of MFCCs to return.
        n_mels : int
            Number of mel bands to generate.
        win_length : float
            Window length in milliseconds.
        overlap : float
            Overlap length in milliseconds.
        fmin : int
            Minimum frequency in mel filterbank in Hz.
        fmax : int
            Maximum frequency in mel filterbank in Hz.
        premphasis : float
            Coefficient for pre-emphasis filter.
        lifter : int
            Liftering coefficient.
        deltas : bool
            Whether to return delta and delta-delta coefficients.
        summarise : bool
            Whether to summarise MFCCs by utterance. If True, returns mean and standard
            deviation of MFCCs for each utterance.

        Returns:
        --------
        features_vec : np.ndarray
            MFCCs for each frame in the audio file. If deltas is True, contains also 
            delta and delta-delta coefficients. Shape is (n_frames, n_mfccs).
            If summarise = True, contains mean and standard deviation of MFCCs for 
            each utterance and shape is (n_mfccs*2,).
        """
        # pre-emphasis filter
        y: np.ndarray = librosa.effects.preemphasis(y, coef=premphasis)

        # compute frame length and overlap in samples
        n_fft: int = int(win_length * sr / 1000)
        hop_length: int = int(overlap * sr / 1000)

        # warn if n_mfcc > n_mels
        if n_mfcc > n_mels:
            warnings.warn(
                f"n_mfcc ({n_mfcc}) is greater than n_mels ({n_mels}). Setting n_mels = n_mfcc.")
            n_mels = n_mfcc

        # extract mfccs
        features_vec: np.ndarray = librosa.feature.mfcc(
            y=y,
            sr=sr,
            n_mfcc=n_mfcc,
            n_mels=n_mels,
            n_fft=n_fft,
            hop_length=hop_length,
            window='hamming',
            lifter=lifter,
            fmin=fmin,
            fmax=fmax
        )

        # delta features
        if deltas:
            features_vec: np.ndarray = self._delta_delta(features_vec)

        # summarise by utterance
        if summarise:
            features_vec: np.ndarray = self._summarise_features(features_vec)

        return features_vec.T

    # Common acoustic features

    def acoustic_features(
        self,
        sound: parselmouth.Sound,
        f0min: float = 75.0,
        f0max: float = 600.0,
    ) -> np.ndarray:
        """
        Extracts acoustic features from audio file.

        Parameters:
        -----------
        sound : parselmouth.Sound
            Parselmouth sound object.
        f0min : float
            Minimum pitch frequency in Hz.
        f0max : float
            Maximum pitch frequency in Hz.

        Returns:
        --------
        features_vec : np.ndarray
            Acoustic features for each frame in the audio file. Shape is (n_features,).
            The extracted features are:
            - pitch (cross-correlation): mean, median, minimum, maximum, standard deviation
            - formants (burg): F1, F2, F3, F4 averages
            - formant dispersion
            - average formant
            - geometric mean of formants
            - fitch vtl
            - delta f
            - hnr
            - jitter
            - shimmer
            - energy (rms)
        """
        # Pitch
        pitch_features: Tuple[float, ...] = self._pitch(
            sound=sound, f0min=f0min, f0max=f0max
        )

        # Formants
        maximum_formant: int
        if pitch_features[0] < 120:
            maximum_formant = 5000
        elif pitch_features[0] >= 120:
            maximum_formant = 5500
        else:
            maximum_formant = 5500
        formants_features: Tuple[float, ...] = self._formants(
            sound=sound, maximum_formant=maximum_formant)

        # Vocal tract estimates
        vt_estimates = self._vocal_tract_estimates(formants=formants_features)

        # HNR
        hnr: float = self._hnr(sound=sound, fmin=f0min)

        # Jitter & Shimmer
        jitter: float
        shimmer: float
        jitter, shimmer = self._jitter_shimmer(sound=sound)

        # RMS energy
        rms_energy: float = self._rms_energy(sound=sound)

        # Concatenate features
        features_vec: np.ndarray = np.concatenate(
            (pitch_features, formants_features, vt_estimates,
             (hnr, jitter, shimmer, rms_energy)), axis=0
        )

        return features_vec

    # Low level features

    def low_lvl_features(
        self,
        y: np.ndarray,
        sr: int,
        win_length: float = 25,
        overlap: float = 10,
        premphasis: float = 0.95,
        n_bands_contrasts: int = 6,
        use_mean_contrasts: bool = False,
        summarise: bool = False
    ) -> np.ndarray:
        """
        Extract low-level features from audio file.

        Parameters:
        -----------
        y : np.ndarray
            Audio time series.
        sr : int
            Sampling rate of y.
        win_length : float
            Window length in milliseconds.
        overlap : float
            Overlap length in milliseconds.
        premphasis : float
            Coefficient for pre-emphasis filter.
        n_bands_contrasts : int
            Number of spectral bands for spectral contrasts.
        use_mean_contrasts : bool
            Take the mean of spectral contrasts over spectral bands. default=False.
        summarise : bool
            Whether to summarise features by utterance. If True, returns mean and standard
            deviation of features for each utterance.

        Returns:
        --------
        features_vec : np.ndarray
            Low-level features for each frame in the audio file. Shape is (n_frames, n_features).
            The extracted features are:
            - spectral centroid
            - spectral bandwidth
            - spectral contrasts
            - spectral flatness
            - spectral rolloff
            - zero crossing rate
        """
        # pre-emphasis filter
        y: np.ndarray = librosa.effects.preemphasis(y, coef=premphasis)

        # compute frame length and overlap in samples
        n_fft: int = int(win_length * sr / 1000)
        hop_length: int = int(overlap * sr / 1000)

        # setting fmin for spectral contrasts
        fmin: float = 200.0

        # Spectral centroid
        spectral_centroids: np.ndarray = librosa.feature.spectral_centroid(
            y=y, sr=sr, n_fft=n_fft, hop_length=hop_length, window='hamming'
        )

        # Spectral bandwidth
        spectral_bandwidths: np.ndarray = librosa.feature.spectral_bandwidth(
            y=y, sr=sr, n_fft=n_fft, hop_length=hop_length, window='hamming'
        )

        # Spectral contrast
        try:
            spectral_contrasts: np.ndarray = librosa.feature.spectral_contrast(
                y=y, sr=sr, n_fft=n_fft, hop_length=hop_length, window='hamming',
                n_bands=n_bands_contrasts, fmin=fmin
            )
        except librosa.util.exceptions.ParameterError:
            fmin: float = 0.5 * sr * 2**(-n_bands_contrasts)
            spectral_contrasts: np.ndarray = librosa.feature.spectral_contrast(
                y=y, sr=sr, n_fft=n_fft, hop_length=hop_length, window='hamming',
                n_bands=n_bands_contrasts, fmin=fmin
            )

        if use_mean_contrasts:
            spectral_contrasts: np.ndarray = np.mean(
                spectral_contrasts, axis=0, keepdims=True
            )

        # Spectral flatness
        spectral_flatness: np.ndarray = librosa.feature.spectral_flatness(
            y=y, n_fft=n_fft, hop_length=hop_length, window='hamming'
        )

        # Spectral rolloff
        spectral_rolloff: np.ndarray = librosa.feature.spectral_rolloff(
            y=y, sr=sr, n_fft=n_fft, hop_length=hop_length, window='hamming'
        )

        # Zero crossing rate
        zero_crossing_rate: np.ndarray = librosa.feature.zero_crossing_rate(
            y=y, frame_length=n_fft, hop_length=hop_length
        )

        # Concatenate features
        features_vec: np.ndarray = np.concatenate(
            (spectral_centroids, spectral_bandwidths, spectral_contrasts,
             spectral_flatness, spectral_rolloff, zero_crossing_rate), axis=0
        )

        # Summarise by utterance
        if summarise:
            features_vec: np.ndarray = self._summarise_features(features_vec)

        return features_vec.T

    # LPCCs
    def lpc_features(
        self,
        y: np.ndarray,
        sr: int,
        n_lpcc: int = 13,
        win_length: float = 25.0,
        overlap: float = 10.0,
        premphasis: float = 0.95,
        order: int = 16,
        summarise: bool = False
    ) -> np.ndarray:
        """
        Extracts LPCCs from audio file by taking the inverse Fourier Transform of the 
        log LP spectrum computed from LPCs (using Burg method).

        Parameters:
        -----------
        audio_file : str
            Path to audio file.
        n_lpcc : int
            Number of LPCCs to return.
        win_length : float
            Window length in milliseconds.
        overlap : float
            Overlap length in milliseconds.
        premphasis : float
            Coefficient for pre-emphasis filter.
        order : int
            Order of the LPC.
        summarise : bool
            Whether to summarise features by utterance. If True, returns mean and standard
            deviation of features for each utterance.

        Returns:
        --------
        features_vec : np.ndarray
            LPCCs for each frame in the audio file. Shape is (n_frames, n_lpccs).
            If summarise = True, shape is (n_lpccs*2,).
        """
        # pre-emphasis filter
        y: np.ndarray = librosa.effects.preemphasis(y, coef=premphasis)

        # warn if n_lpcc > order
        if n_lpcc > order:
            warnings.warn(
                f"n_lpcc ({n_lpcc}) is greater than order ({order}). Setting order = n_lpcc.")
            order = n_lpcc

        # compute frame length and overlap in samples
        n_fft: int = int(win_length * sr / 1000)
        hop_length: int = int(overlap * sr / 1000)

        # Frame signal
        frames: np.ndarray = librosa.util.frame(
            y, frame_length=n_fft, hop_length=hop_length
        )

        # Windowing
        windowed_frames: np.ndarray = frames * \
            signal.windows.hamming(n_fft)[:, None]

        # Warn if num samples per frame < order
        if windowed_frames.shape[0] < order:
            warnings.warn(
                f"Number of samples per frame ({windowed_frames.shape[0]}) is less than order ({order}). Setting order = num samples per frame - 1.")
            order = windowed_frames.shape[0] - 1

            if n_lpcc > order:
                warnings.warn(
                    f"n_lpcc ({n_lpcc}) is greater than order ({order}) but order was greater than num samples per frame. Some LPCCs will be set to 0.")

        # Compute LPCs
        lpcs: np.ndarray = np.array([librosa.lpc(y=frame, order=order)
                                    for frame in windowed_frames.T])

        # Compute LP spectrum
        lp_spectrum: np.ndarray = np.abs(np.fft.rfft(lpcs, axis=1))

        # Compute LPCCs
        features_vec: np.ndarray = np.fft.irfft(
            np.log(lp_spectrum), axis=1)[:, :n_lpcc]

        # Summarise by utterance
        if summarise:
            features_vec: np.ndarray = self._summarise_features(features_vec.T)

        return features_vec

    # Metatdata extraction

    def extract_metadata(self, filename: str, metavars: list = None, separator: str = '_', add_selection_column: bool = False) -> dict:
        """
        Extracts metadata variables from the filename.

        Parameters:
        -----------
        filename: str
            Filename.
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
        # Create empty dictionary
        metadata_dict = {}

        # Extract metadata from filename
        basename = os.path.basename(filename)
        if metavars:
            metadata = os.path.splitext(basename)[0]
            metadata = metadata.split(separator)

        # append to dictionary a dictionary of metadata
        metadata_dict['filename'] = [basename]
        if metavars:
            for i, var in enumerate(metavars):
                if var != '-':
                    metadata_dict[var] = [metadata[i]]

        # Add selection column
        if add_selection_column:
            metadata_dict['selection'] = [0]

        return metadata_dict

    def extract_feature_labels(self, feature_methods: dict) -> list:
        """
        Extracts feature labels from feature_methods.

        Parameters:
        -----------
        feature_methods : dict
            Dictionary of feature extraction methods to use. Keys are the names of the
            feature extraction methods, values are the arguments to pass to the methods.
            Methods are: 'mel_features', 'acoustic_features', 'low_lvl_features', 'lpc_features'.

        Returns:
        --------
        list
            List of feature labels.
        """
        feature_labels: list = []
        for method in feature_methods:
            if method == "mel_features":
                tmp_feature_labels: list = [f'mfcc_{i+1}' for i in range(
                    feature_methods[method]['n_mfcc'])]
                if feature_methods[method]['deltas']:
                    tmp_feature_labels.extend(
                        [f'delta_{i}' for i in range(feature_methods[method]['n_mfcc'])] +
                        [f'delta_delta_{i}' for i in range(
                            feature_methods[method]['n_mfcc'])]
                    )
                if feature_methods[method]['summarise']:
                    feature_labels.extend(
                        [f'{label}_mean' for label in tmp_feature_labels] +
                        [f'{label}_std' for label in tmp_feature_labels]
                    )
                else:
                    feature_labels.extend(tmp_feature_labels)

            elif method == "acoustic_features":
                feature_labels.extend([
                    'pitch_mean', 'pitch_median', 'pitch_min', 'pitch_max', 'pitch_std',
                    'f1', 'f2', 'f3', 'f4',
                    'formant_dispersion', 'avg_formant', 'geometric_mean_formants', 'fitch_vtl', 'delta_f',
                    'hnr', 'jitter', 'shimmer', 'rms_energy'
                ])

            elif method == "low_lvl_features":
                if feature_methods[method]['use_mean_contrasts']:
                    tmp_feature_labels: list = [
                        'spectral_centroid', 'spectral_bandwidth', 'spectral_contrast',
                        'spectral_flatness', 'spectral_rolloff', 'zero_crossing_rate'
                    ]
                else:
                    tmp_feature_labels: list = ['spectral_centroid', 'spectral_bandwidth'] + \
                        [f'spectral_contrast_{i+1}' for i in range(
                            feature_methods[method]['n_bands_contrasts']+1)] + \
                        ['spectral_flatness', 'spectral_rolloff', 'zero_crossing_rate']
                if feature_methods[method]['summarise']:
                    feature_labels.extend(
                        [f'{label}_mean' for label in tmp_feature_labels] +
                        [f'{label}_std' for label in tmp_feature_labels]
                    )
                else:
                    feature_labels.extend(tmp_feature_labels)

            elif method == "lpc_features":
                tmp_feature_labels: list = [f'lpcc_{i}' for i in range(
                    feature_methods[method]['n_lpcc'])]
                if feature_methods[method]['summarise']:
                    feature_labels.extend(
                        [f'{label}_mean' for label in tmp_feature_labels] +
                        [f'{label}_std' for label in tmp_feature_labels]
                    )
                else:
                    feature_labels.extend(tmp_feature_labels)

        return feature_labels

    # Process sound files
    def process_files(self):
        """
        Processes each file with the specified feature extraction methods.

        Returns:
        --------
        features : np.ndarray
            Extracted features. Shape is (n_files, n_features) when summarise=True for 
            mel_features, low_lvl_features, lpc_features. If summarise=False, shape is (n_files x n_frames, n_features).
        metadata_values : np.ndarray
            Extracted metadata. Shape is (n_files x n_frames, n_metavars).
        metadata_labels : list
            List of metadata variable names.
        """
        features: list = []  # list of features of all files
        metadata_dict: defaultdict = defaultdict(
            list)  # dictionary of metadata
        feature_labels: list = self.extract_feature_labels(
            feature_methods=self.feature_methods)  # list of feature labels

        for audio_file in self.audio_files:
            if any(method in self.feature_methods for method in ['mel_features', 'lpc_features', 'low_lvl_features']):
                y: np.ndarray
                sr: int
                y, sr = librosa.load(audio_file, sr=None)

            if 'acoustic_features' in self.feature_methods:
                sound: parselmouth.Sound = parselmouth.Sound(audio_file)

            # Extract metadata
            metadata_dict_file: dict = self.extract_metadata(
                filename=audio_file, **self.metadata_vars
            )

            file_feature: list = []  # list of features of current file
            for method, args in self.feature_methods.items():
                if method == 'mel_features':
                    mel_features_vec: np.ndarray
                    mel_features_vec = self.mel_features(
                        y=y, sr=sr, **args)
                    file_feature.append(mel_features_vec)
                elif method == 'acoustic_features':
                    ac_features_vec: np.ndarray
                    ac_features_vec = self.acoustic_features(
                        sound=sound, **args)
                    file_feature.append(ac_features_vec)
                elif method == 'low_lvl_features':
                    low_lvl_features_vec: np.ndarray
                    low_lvl_features_vec = self.low_lvl_features(
                        y=y, sr=sr, **args)
                    file_feature.append(low_lvl_features_vec)
                elif method == 'lpc_features':
                    lpc_features_vec: np.ndarray
                    lpc_features_vec = self.lpc_features(
                        y=y, sr=sr, **args)
                    file_feature.append(lpc_features_vec)

            # Concatenate features of current file and append to features
            file_feature: np.ndarray = np.concatenate(file_feature, axis=0)
            # Reshape if format is incorrect
            if len(file_feature.shape) == 1:
                file_feature = file_feature.reshape(1, -1)
            features.append(file_feature)

            # Concatenate metadata of current file to the samed 0th dim of features and append
            for key, value in metadata_dict_file.items():
                metadata_dict[key].extend(value*file_feature.shape[0])

        # Split metadata into labels and values
        metadata_labels: list = list(metadata_dict.keys())
        metadata_values: np.ndarray = np.array(list(metadata_dict.values())).T

        return np.concatenate(features, axis=0), metadata_values, metadata_labels, feature_labels


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
    fe = FeatureExtractor(audio_dir, feature_methods, metadata_vars)
    features, metavalues, metalabels = fe.process_files()
    print(features.shape)
    print(metavalues)
    print(metalabels)
