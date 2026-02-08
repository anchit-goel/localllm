"""
Preprocessing utilities for audio and image data.
"""

import numpy as np
from typing import Optional, Tuple, Union
import logging

logger = logging.getLogger(__name__)


class AudioPreprocessor:
    """Audio preprocessing utilities."""
    
    def __init__(self, sample_rate: int = 16000):
        self.sample_rate = sample_rate
    
    def normalize(self, waveform: np.ndarray) -> np.ndarray:
        """Normalize audio to [-1, 1] range."""
        max_val = np.max(np.abs(waveform))
        if max_val > 0:
            return waveform / max_val
        return waveform
    
    def trim_silence(
        self,
        waveform: np.ndarray,
        threshold_db: float = -40,
    ) -> np.ndarray:
        """Trim leading and trailing silence."""
        try:
            import librosa
            trimmed, _ = librosa.effects.trim(waveform, top_db=abs(threshold_db))
            return trimmed
        except:
            return waveform
    
    def resample(
        self,
        waveform: np.ndarray,
        orig_sr: int,
        target_sr: int,
    ) -> np.ndarray:
        """Resample audio to target sample rate."""
        try:
            import librosa
            return librosa.resample(waveform, orig_sr=orig_sr, target_sr=target_sr)
        except:
            return waveform
    
    def augment(
        self,
        waveform: np.ndarray,
        noise_level: float = 0.005,
        time_stretch: Optional[float] = None,
        pitch_shift: Optional[int] = None,
    ) -> np.ndarray:
        """Apply augmentation for training."""
        result = waveform.copy()
        
        # Add noise
        if noise_level > 0:
            noise = np.random.randn(len(result)) * noise_level
            result = result + noise
        
        try:
            import librosa
            
            if time_stretch is not None:
                result = librosa.effects.time_stretch(result, rate=time_stretch)
            
            if pitch_shift is not None:
                result = librosa.effects.pitch_shift(
                    result, sr=self.sample_rate, n_steps=pitch_shift
                )
        except:
            pass
        
        return result


class ImagePreprocessor:
    """Image preprocessing utilities."""
    
    def __init__(self, target_size: Tuple[int, int] = (224, 224)):
        self.target_size = target_size
    
    def normalize(self, image: np.ndarray) -> np.ndarray:
        """Normalize image to [0, 1] range."""
        return image.astype(np.float32) / 255.0
    
    def resize(self, image: np.ndarray) -> np.ndarray:
        """Resize image to target size."""
        try:
            import cv2
            return cv2.resize(image, self.target_size)
        except:
            return image
    
    def to_grayscale(self, image: np.ndarray) -> np.ndarray:
        """Convert to grayscale."""
        try:
            import cv2
            if len(image.shape) == 3:
                return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            return image
        except:
            return image
    
    def enhance_contrast(self, image: np.ndarray) -> np.ndarray:
        """Enhance image contrast using CLAHE."""
        try:
            import cv2
            if len(image.shape) == 3:
                lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
                l, a, b = cv2.split(lab)
                clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
                l = clahe.apply(l)
                lab = cv2.merge([l, a, b])
                return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
            else:
                clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
                return clahe.apply(image)
        except:
            return image
