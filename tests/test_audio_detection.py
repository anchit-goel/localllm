"""
Tests for audio detection module.
"""

import pytest
import numpy as np
import tempfile
import os

# Test utilities
def generate_test_audio(duration: float = 1.0, sample_rate: int = 16000) -> np.ndarray:
    """Generate test audio waveform."""
    t = np.linspace(0, duration, int(sample_rate * duration), endpoint=False)
    # Generate a simple sine wave with harmonics
    waveform = 0.5 * np.sin(2 * np.pi * 440 * t)  # A4 note
    waveform += 0.3 * np.sin(2 * np.pi * 880 * t)  # Harmonic
    waveform += 0.1 * np.sin(2 * np.pi * 1320 * t)  # Another harmonic
    # Add some noise
    waveform += 0.05 * np.random.randn(len(waveform))
    return waveform.astype(np.float32)


class TestAudioFeatureExtraction:
    """Tests for audio feature extraction."""
    
    def test_mfcc_extraction_shape(self):
        """Test MFCC extraction produces correct shape."""
        from src.audio.feature_extraction import AudioFeatureExtractor
        
        extractor = AudioFeatureExtractor(n_mfcc=40)
        waveform = generate_test_audio(duration=2.0)
        
        mfcc, delta, delta2 = extractor.extract_mfcc(waveform)
        
        assert mfcc.shape[0] == 40, "MFCC should have 40 coefficients"
        assert delta.shape == mfcc.shape, "Delta should match MFCC shape"
        assert delta2.shape == mfcc.shape, "Delta2 should match MFCC shape"
    
    def test_spectral_features(self):
        """Test spectral feature extraction."""
        from src.audio.feature_extraction import AudioFeatureExtractor
        
        extractor = AudioFeatureExtractor()
        waveform = generate_test_audio()
        
        features = extractor.extract_spectral_features(waveform)
        
        assert 'spectral_centroid' in features
        assert 'spectral_rolloff' in features
        assert 'spectral_bandwidth' in features
        assert 'spectral_contrast' in features
        assert 'spectral_flatness' in features
    
    def test_mel_spectrogram_shape(self):
        """Test mel spectrogram generation."""
        from src.audio.feature_extraction import AudioFeatureExtractor
        
        extractor = AudioFeatureExtractor(n_mels=128)
        waveform = generate_test_audio()
        
        mel_spec, mel_spec_db = extractor.extract_mel_spectrogram(waveform)
        
        assert mel_spec.shape[0] == 128, "Should have 128 mel bands"
        assert mel_spec_db.shape == mel_spec.shape
    
    def test_all_features_extraction(self):
        """Test complete feature extraction pipeline."""
        from src.audio.feature_extraction import AudioFeatureExtractor
        
        extractor = AudioFeatureExtractor()
        waveform = generate_test_audio(duration=3.0)
        
        features = extractor.extract_all_features(waveform, sample_rate=16000)
        
        assert features.mfcc is not None
        assert features.mel_spectrogram is not None
        assert features.chroma is not None
        assert features.duration > 0
    
    def test_cnn_input_preparation(self):
        """Test CNN input preparation with padding/cropping."""
        from src.audio.feature_extraction import AudioFeatureExtractor
        
        extractor = AudioFeatureExtractor()
        waveform = generate_test_audio()
        
        features = extractor.extract_all_features(waveform, sample_rate=16000)
        cnn_input = extractor.prepare_cnn_input(features.mel_spectrogram_db, target_length=128)
        
        assert cnn_input.shape == (128, 128, 1), f"Expected (128, 128, 1), got {cnn_input.shape}"
    
    def test_concatenated_features(self):
        """Test feature concatenation for ML models."""
        from src.audio.feature_extraction import AudioFeatureExtractor
        
        extractor = AudioFeatureExtractor()
        waveform = generate_test_audio()
        
        features = extractor.extract_all_features(waveform, sample_rate=16000)
        concat = features.get_concatenated_features()
        
        assert len(concat.shape) == 1, "Should be 1D vector"
        assert len(concat) > 100, "Should have many features"


class TestDeepfakeDetector:
    """Tests for deepfake detection."""
    
    def test_detector_initialization(self):
        """Test detector initializes correctly."""
        from src.audio.deepfake_detector import DeepfakeDetector
        
        detector = DeepfakeDetector()
        
        assert detector.feature_extractor is not None
        assert detector.cnn_model is not None
        assert detector.lstm_model is not None
        assert detector.ensemble_model is not None
    
    def test_detection_on_waveform(self):
        """Test detection on numpy waveform."""
        from src.audio.deepfake_detector import DeepfakeDetector
        
        detector = DeepfakeDetector()
        waveform = generate_test_audio(duration=3.0)
        
        result = detector.detect(waveform, sample_rate=16000)
        
        assert 0 <= result.risk_score <= 1
        assert isinstance(result.is_fake, bool)
        assert result.explanation is not None
    
    def test_risk_level_classification(self):
        """Test risk level classification."""
        from src.audio.deepfake_detector import DeepfakeDetector
        
        detector = DeepfakeDetector()
        
        assert detector.get_risk_level(0.1) == "PASS"
        assert detector.get_risk_level(0.4) == "FLAG"
        assert detector.get_risk_level(0.7) == "BLOCK"
        assert detector.get_risk_level(0.9) == "ALERT"
    
    def test_result_to_dict(self):
        """Test result serialization."""
        from src.audio.deepfake_detector import DeepfakeDetector
        
        detector = DeepfakeDetector()
        waveform = generate_test_audio()
        
        result = detector.detect(waveform, sample_rate=16000)
        result_dict = result.to_dict()
        
        assert 'risk_score' in result_dict
        assert 'is_fake' in result_dict
        assert 'explanation' in result_dict


class TestVoiceCloneDetector:
    """Tests for voice clone detection."""
    
    def test_detector_initialization(self):
        """Test detector initializes correctly."""
        from src.audio.voice_clone_detector import VoiceCloneDetector
        
        detector = VoiceCloneDetector()
        
        assert detector.embedder is not None
        assert detector.voice_db is not None
    
    def test_detection_on_waveform(self):
        """Test detection on numpy waveform."""
        from src.audio.voice_clone_detector import VoiceCloneDetector
        
        detector = VoiceCloneDetector()
        waveform = generate_test_audio(duration=3.0)
        
        result = detector.detect(waveform, sample_rate=16000)
        
        assert 0 <= result.similarity_score <= 1
        assert isinstance(result.is_cloned, bool)
        assert result.embedding is not None
    
    def test_speaker_registration(self):
        """Test speaker registration and verification."""
        from src.audio.voice_clone_detector import VoiceCloneDetector
        
        detector = VoiceCloneDetector()
        
        # Register speaker
        waveform1 = generate_test_audio(duration=3.0)
        waveform2 = generate_test_audio(duration=3.0)
        
        detector.register_speaker("test_speaker", [waveform1, waveform2])
        
        assert "test_speaker" in detector.voice_db.embeddings


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
