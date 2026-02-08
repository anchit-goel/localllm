"""
Audio Feature Extraction Pipeline.

Extracts comprehensive audio features for deepfake detection including:
- MFCC (Mel-Frequency Cepstral Coefficients)
- Spectral features (centroid, rolloff, bandwidth, contrast)
- Chroma features for harmonic content
- Zero-crossing rate and RMS energy
- Delta and delta-delta features for temporal dynamics
- Mel-spectrogram generation for CNN input
"""

import numpy as np
from typing import Dict, Tuple, Optional, Union
from dataclasses import dataclass
import logging

try:
    import librosa
    import librosa.display
except ImportError:
    librosa = None

try:
    from scipy import signal
    from scipy.fft import fft, fftfreq
except ImportError:
    signal = None
    fft = None
    fftfreq = None

logger = logging.getLogger(__name__)


@dataclass
class AudioFeatures:
    """Container for extracted audio features."""
    mfcc: np.ndarray
    mfcc_delta: np.ndarray
    mfcc_delta2: np.ndarray
    spectral_centroid: np.ndarray
    spectral_rolloff: np.ndarray
    spectral_bandwidth: np.ndarray
    spectral_contrast: np.ndarray
    spectral_flatness: np.ndarray
    chroma: np.ndarray
    zero_crossing_rate: np.ndarray
    rms_energy: np.ndarray
    mel_spectrogram: np.ndarray
    mel_spectrogram_db: np.ndarray
    waveform: np.ndarray
    sample_rate: int
    duration: float
    
    def to_dict(self) -> Dict[str, np.ndarray]:
        """Convert features to dictionary."""
        return {
            'mfcc': self.mfcc,
            'mfcc_delta': self.mfcc_delta,
            'mfcc_delta2': self.mfcc_delta2,
            'spectral_centroid': self.spectral_centroid,
            'spectral_rolloff': self.spectral_rolloff,
            'spectral_bandwidth': self.spectral_bandwidth,
            'spectral_contrast': self.spectral_contrast,
            'spectral_flatness': self.spectral_flatness,
            'chroma': self.chroma,
            'zero_crossing_rate': self.zero_crossing_rate,
            'rms_energy': self.rms_energy,
            'mel_spectrogram': self.mel_spectrogram,
            'mel_spectrogram_db': self.mel_spectrogram_db,
        }
    
    def get_concatenated_features(self) -> np.ndarray:
        """Get all features concatenated for traditional ML models."""
        features = []
        
        # Aggregate time-varying features using statistics
        for feat in [self.mfcc, self.mfcc_delta, self.mfcc_delta2,
                     self.spectral_centroid, self.spectral_rolloff,
                     self.spectral_bandwidth, self.spectral_contrast,
                     self.spectral_flatness, self.chroma,
                     self.zero_crossing_rate, self.rms_energy]:
            if feat is not None and len(feat.shape) > 0:
                if len(feat.shape) == 1:
                    feat = feat.reshape(1, -1)
                features.extend([
                    np.mean(feat, axis=1),
                    np.std(feat, axis=1),
                    np.min(feat, axis=1),
                    np.max(feat, axis=1),
                    np.median(feat, axis=1),
                ])
        
        return np.concatenate(features)


class AudioFeatureExtractor:
    """
    Comprehensive audio feature extraction for deepfake detection.
    
    Attributes:
        sample_rate: Target sample rate for audio processing (default: 16000 Hz)
        n_mfcc: Number of MFCC coefficients to extract (default: 40)
        n_mels: Number of mel bands for spectrogram (default: 128)
        n_fft: FFT window size (default: 2048)
        hop_length: Hop length for STFT (default: 512)
    """
    
    def __init__(
        self,
        sample_rate: int = 16000,
        n_mfcc: int = 40,
        n_mels: int = 128,
        n_fft: int = 2048,
        hop_length: int = 512,
    ):
        if librosa is None:
            raise ImportError("librosa is required for audio feature extraction. Install with: pip install librosa")
        
        self.sample_rate = sample_rate
        self.n_mfcc = n_mfcc
        self.n_mels = n_mels
        self.n_fft = n_fft
        self.hop_length = hop_length
        
        logger.info(f"AudioFeatureExtractor initialized with sr={sample_rate}, n_mfcc={n_mfcc}, n_mels={n_mels}")
    
    def load_audio(
        self,
        audio_path: str,
        duration: Optional[float] = None,
        offset: float = 0.0,
    ) -> Tuple[np.ndarray, int]:
        """
        Load audio file and resample to target sample rate.
        
        Args:
            audio_path: Path to audio file
            duration: Duration to load (None for full file)
            offset: Start position in seconds
            
        Returns:
            Tuple of (waveform, sample_rate)
        """
        waveform, sr = librosa.load(
            audio_path,
            sr=self.sample_rate,
            duration=duration,
            offset=offset,
            mono=True,
        )
        
        logger.debug(f"Loaded audio: {len(waveform)/sr:.2f}s at {sr}Hz")
        return waveform, sr
    
    def load_audio_from_bytes(self, audio_bytes: bytes) -> Tuple[np.ndarray, int]:
        """
        Load audio from bytes (for API usage).
        
        Args:
            audio_bytes: Audio file as bytes
            
        Returns:
            Tuple of (waveform, sample_rate)
        """
        import io
        import soundfile as sf
        
        with io.BytesIO(audio_bytes) as f:
            waveform, sr = sf.read(f)
            
        # Convert to mono if stereo
        if len(waveform.shape) > 1:
            waveform = np.mean(waveform, axis=1)
        
        # Resample if necessary
        if sr != self.sample_rate:
            waveform = librosa.resample(waveform, orig_sr=sr, target_sr=self.sample_rate)
            sr = self.sample_rate
            
        return waveform, sr
    
    def extract_mfcc(self, waveform: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Extract MFCC features with delta and delta-delta.
        
        Args:
            waveform: Audio waveform
            
        Returns:
            Tuple of (mfcc, delta, delta2)
        """
        mfcc = librosa.feature.mfcc(
            y=waveform,
            sr=self.sample_rate,
            n_mfcc=self.n_mfcc,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
        )
        
        # Compute deltas for temporal dynamics
        mfcc_delta = librosa.feature.delta(mfcc, order=1)
        mfcc_delta2 = librosa.feature.delta(mfcc, order=2)
        
        return mfcc, mfcc_delta, mfcc_delta2
    
    def extract_spectral_features(self, waveform: np.ndarray) -> Dict[str, np.ndarray]:
        """
        Extract spectral features.
        
        Args:
            waveform: Audio waveform
            
        Returns:
            Dictionary of spectral features
        """
        # Spectral centroid - "brightness" of sound
        spectral_centroid = librosa.feature.spectral_centroid(
            y=waveform, sr=self.sample_rate,
            n_fft=self.n_fft, hop_length=self.hop_length
        )
        
        # Spectral rolloff - frequency below which 85% of energy lies
        spectral_rolloff = librosa.feature.spectral_rolloff(
            y=waveform, sr=self.sample_rate,
            n_fft=self.n_fft, hop_length=self.hop_length,
            roll_percent=0.85
        )
        
        # Spectral bandwidth - range of frequencies
        spectral_bandwidth = librosa.feature.spectral_bandwidth(
            y=waveform, sr=self.sample_rate,
            n_fft=self.n_fft, hop_length=self.hop_length
        )
        
        # Spectral contrast - difference between peaks and valleys
        spectral_contrast = librosa.feature.spectral_contrast(
            y=waveform, sr=self.sample_rate,
            n_fft=self.n_fft, hop_length=self.hop_length
        )
        
        # Spectral flatness - "noisiness" of signal
        spectral_flatness = librosa.feature.spectral_flatness(
            y=waveform,
            n_fft=self.n_fft, hop_length=self.hop_length
        )
        
        return {
            'spectral_centroid': spectral_centroid,
            'spectral_rolloff': spectral_rolloff,
            'spectral_bandwidth': spectral_bandwidth,
            'spectral_contrast': spectral_contrast,
            'spectral_flatness': spectral_flatness,
        }
    
    def extract_chroma(self, waveform: np.ndarray) -> np.ndarray:
        """
        Extract chroma features for harmonic content analysis.
        
        Args:
            waveform: Audio waveform
            
        Returns:
            Chroma features (12 x time_frames)
        """
        chroma = librosa.feature.chroma_stft(
            y=waveform, sr=self.sample_rate,
            n_fft=self.n_fft, hop_length=self.hop_length
        )
        return chroma
    
    def extract_temporal_features(self, waveform: np.ndarray) -> Dict[str, np.ndarray]:
        """
        Extract temporal domain features.
        
        Args:
            waveform: Audio waveform
            
        Returns:
            Dictionary with ZCR and RMS energy
        """
        # Zero crossing rate - frequency of sign changes
        zcr = librosa.feature.zero_crossing_rate(
            y=waveform,
            frame_length=self.n_fft,
            hop_length=self.hop_length
        )
        
        # RMS energy - loudness approximation
        rms = librosa.feature.rms(
            y=waveform,
            frame_length=self.n_fft,
            hop_length=self.hop_length
        )
        
        return {
            'zero_crossing_rate': zcr,
            'rms_energy': rms,
        }
    
    def extract_mel_spectrogram(self, waveform: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generate mel-spectrogram for CNN input.
        
        Args:
            waveform: Audio waveform
            
        Returns:
            Tuple of (mel_spectrogram, mel_spectrogram_db)
        """
        mel_spec = librosa.feature.melspectrogram(
            y=waveform, sr=self.sample_rate,
            n_fft=self.n_fft, hop_length=self.hop_length,
            n_mels=self.n_mels
        )
        
        # Convert to dB scale for better visualization and model input
        mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
        
        return mel_spec, mel_spec_db
    
    def extract_fft_features(self, waveform: np.ndarray) -> Dict[str, np.ndarray]:
        """
        Extract FFT-based features for artifact detection.
        
        Args:
            waveform: Audio waveform
            
        Returns:
            Dictionary with FFT magnitude and phase
        """
        if fft is None:
            logger.warning("scipy not available for FFT analysis")
            return {'fft_magnitude': np.array([]), 'fft_phase': np.array([])}
        
        # Compute FFT
        n = len(waveform)
        fft_vals = fft(waveform)
        freqs = fftfreq(n, 1/self.sample_rate)
        
        # Get positive frequencies only
        positive_mask = freqs >= 0
        fft_magnitude = np.abs(fft_vals[positive_mask])
        fft_phase = np.angle(fft_vals[positive_mask])
        fft_freqs = freqs[positive_mask]
        
        return {
            'fft_magnitude': fft_magnitude,
            'fft_phase': fft_phase,
            'fft_frequencies': fft_freqs,
        }
    
    def detect_high_frequency_artifacts(self, waveform: np.ndarray) -> Dict[str, float]:
        """
        Detect high-frequency artifacts typical in GAN-generated audio.
        
        Args:
            waveform: Audio waveform
            
        Returns:
            Dictionary with artifact scores
        """
        fft_features = self.extract_fft_features(waveform)
        
        if len(fft_features['fft_magnitude']) == 0:
            return {'high_freq_ratio': 0.0, 'artifact_score': 0.0}
        
        freqs = fft_features['fft_frequencies']
        magnitude = fft_features['fft_magnitude']
        
        # Analyze high frequency content (above 7kHz)
        high_freq_mask = freqs > 7000
        low_freq_mask = (freqs > 100) & (freqs <= 7000)
        
        if np.sum(high_freq_mask) == 0 or np.sum(low_freq_mask) == 0:
            return {'high_freq_ratio': 0.0, 'artifact_score': 0.0}
        
        high_freq_energy = np.sum(magnitude[high_freq_mask] ** 2)
        low_freq_energy = np.sum(magnitude[low_freq_mask] ** 2)
        
        high_freq_ratio = high_freq_energy / (low_freq_energy + 1e-10)
        
        # Detect unnatural patterns in high frequencies
        # GAN audio often has specific artifacts in high frequency bands
        high_freq_variance = np.var(magnitude[high_freq_mask])
        high_freq_mean = np.mean(magnitude[high_freq_mask])
        
        # Coefficient of variation for artifact detection
        cv = high_freq_variance / (high_freq_mean + 1e-10)
        
        # Normalize to 0-1 score
        artifact_score = min(1.0, cv / 10.0)
        
        return {
            'high_freq_ratio': float(high_freq_ratio),
            'artifact_score': float(artifact_score),
            'high_freq_variance': float(high_freq_variance),
        }
    
    def detect_phase_inconsistencies(self, waveform: np.ndarray) -> Dict[str, float]:
        """
        Analyze phase for inconsistencies typical in synthetic audio.
        
        Args:
            waveform: Audio waveform
            
        Returns:
            Dictionary with phase analysis results
        """
        fft_features = self.extract_fft_features(waveform)
        
        if len(fft_features['fft_phase']) == 0:
            return {'phase_variance': 0.0, 'phase_discontinuity_score': 0.0}
        
        phase = fft_features['fft_phase']
        
        # Unwrap phase to detect discontinuities
        unwrapped_phase = np.unwrap(phase)
        
        # Calculate phase derivatives
        phase_diff = np.diff(unwrapped_phase)
        
        # Detect sudden phase jumps (typical in spliced or synthetic audio)
        phase_jumps = np.abs(phase_diff) > np.pi / 2
        phase_discontinuity_score = np.sum(phase_jumps) / (len(phase_jumps) + 1e-10)
        
        # Overall phase variance
        phase_variance = np.var(phase)
        
        return {
            'phase_variance': float(phase_variance),
            'phase_discontinuity_score': float(phase_discontinuity_score),
        }
    
    def extract_all_features(
        self,
        audio_input: Union[str, bytes, np.ndarray],
        sample_rate: Optional[int] = None,
    ) -> AudioFeatures:
        """
        Extract all audio features from input.
        
        Args:
            audio_input: Path to audio file, bytes, or waveform array
            sample_rate: Sample rate if providing waveform array
            
        Returns:
            AudioFeatures object containing all extracted features
        """
        # Load audio based on input type
        if isinstance(audio_input, str):
            waveform, sr = self.load_audio(audio_input)
        elif isinstance(audio_input, bytes):
            waveform, sr = self.load_audio_from_bytes(audio_input)
        elif isinstance(audio_input, np.ndarray):
            waveform = audio_input
            sr = sample_rate or self.sample_rate
            if sr != self.sample_rate:
                waveform = librosa.resample(waveform, orig_sr=sr, target_sr=self.sample_rate)
                sr = self.sample_rate
        else:
            raise ValueError(f"Unsupported audio input type: {type(audio_input)}")
        
        # Extract MFCC features
        mfcc, mfcc_delta, mfcc_delta2 = self.extract_mfcc(waveform)
        
        # Extract spectral features
        spectral = self.extract_spectral_features(waveform)
        
        # Extract chroma features
        chroma = self.extract_chroma(waveform)
        
        # Extract temporal features
        temporal = self.extract_temporal_features(waveform)
        
        # Extract mel spectrogram
        mel_spec, mel_spec_db = self.extract_mel_spectrogram(waveform)
        
        # Calculate duration
        duration = len(waveform) / sr
        
        return AudioFeatures(
            mfcc=mfcc,
            mfcc_delta=mfcc_delta,
            mfcc_delta2=mfcc_delta2,
            spectral_centroid=spectral['spectral_centroid'],
            spectral_rolloff=spectral['spectral_rolloff'],
            spectral_bandwidth=spectral['spectral_bandwidth'],
            spectral_contrast=spectral['spectral_contrast'],
            spectral_flatness=spectral['spectral_flatness'],
            chroma=chroma,
            zero_crossing_rate=temporal['zero_crossing_rate'],
            rms_energy=temporal['rms_energy'],
            mel_spectrogram=mel_spec,
            mel_spectrogram_db=mel_spec_db,
            waveform=waveform,
            sample_rate=sr,
            duration=duration,
        )
    
    def prepare_cnn_input(
        self,
        mel_spectrogram_db: np.ndarray,
        target_length: int = 128,
    ) -> np.ndarray:
        """
        Prepare mel-spectrogram for CNN input with fixed dimensions.
        
        Args:
            mel_spectrogram_db: Mel-spectrogram in dB scale
            target_length: Target number of time frames
            
        Returns:
            Padded/cropped spectrogram of shape (n_mels, target_length, 1)
        """
        current_length = mel_spectrogram_db.shape[1]
        
        if current_length < target_length:
            # Pad with minimum value
            padding = target_length - current_length
            pad_value = mel_spectrogram_db.min()
            mel_spectrogram_db = np.pad(
                mel_spectrogram_db,
                ((0, 0), (0, padding)),
                mode='constant',
                constant_values=pad_value
            )
        elif current_length > target_length:
            # Center crop
            start = (current_length - target_length) // 2
            mel_spectrogram_db = mel_spectrogram_db[:, start:start + target_length]
        
        # Add channel dimension for CNN
        return mel_spectrogram_db[..., np.newaxis]
    
    def prepare_lstm_input(
        self,
        mfcc: np.ndarray,
        sequence_length: int = 100,
    ) -> np.ndarray:
        """
        Prepare MFCC sequence for LSTM input.
        
        Args:
            mfcc: MFCC features (n_mfcc, time_frames)
            sequence_length: Target sequence length
            
        Returns:
            Sequence of shape (sequence_length, n_mfcc)
        """
        # Transpose to (time_frames, n_mfcc)
        mfcc_transposed = mfcc.T
        current_length = mfcc_transposed.shape[0]
        
        if current_length < sequence_length:
            # Pad with zeros
            padding = sequence_length - current_length
            mfcc_transposed = np.pad(
                mfcc_transposed,
                ((0, padding), (0, 0)),
                mode='constant',
                constant_values=0
            )
        elif current_length > sequence_length:
            # Center crop
            start = (current_length - sequence_length) // 2
            mfcc_transposed = mfcc_transposed[start:start + sequence_length]
        
        return mfcc_transposed
