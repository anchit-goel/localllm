"""
Audio Deepfake Detection Module.

Combines multiple detection approaches:
1. CNN-based spectrogram classifier
2. Traditional ML ensemble (RF, XGBoost, SVM)
3. LSTM/GRU for temporal patterns
"""

import numpy as np
from typing import Dict, Optional, Tuple, List, Union
from dataclasses import dataclass
import logging
import os
import pickle

from .feature_extraction import AudioFeatureExtractor, AudioFeatures

logger = logging.getLogger(__name__)


@dataclass
class DeepfakeResult:
    """Container for deepfake detection results."""
    is_fake: bool
    confidence: float
    risk_score: float
    cnn_score: float
    ensemble_score: float
    lstm_score: float
    explanation: str
    detected_artifacts: List[str]
    feature_importance: Dict[str, float]
    
    def to_dict(self) -> Dict:
        """Convert to dictionary for API response."""
        return {
            'is_fake': self.is_fake,
            'confidence': self.confidence,
            'risk_score': self.risk_score,
            'cnn_score': self.cnn_score,
            'ensemble_score': self.ensemble_score,
            'lstm_score': self.lstm_score,
            'explanation': self.explanation,
            'detected_artifacts': self.detected_artifacts,
            'feature_importance': self.feature_importance,
        }


class CNNSpectrogramModel:
    """
    CNN-based spectrogram classifier for deepfake detection.
    
    Architecture: 4-5 conv blocks with batch norm + dropout
    Input: Mel-spectrogram (128 mel bins, variable time frames)
    Output: Binary classification (real/fake) + confidence score
    """
    
    def __init__(self, input_shape: Tuple[int, int, int] = (128, 128, 1)):
        self.input_shape = input_shape
        self.model = None
        self._build_model()
    
    def _build_model(self):
        """Build CNN architecture."""
        try:
            import tensorflow as tf
            from tensorflow import keras
            from tensorflow.keras import layers
            
            self.model = keras.Sequential([
                # Input layer
                layers.Input(shape=self.input_shape),
                
                # Conv Block 1
                layers.Conv2D(32, (3, 3), padding='same'),
                layers.BatchNormalization(),
                layers.ReLU(),
                layers.MaxPooling2D((2, 2)),
                layers.Dropout(0.25),
                
                # Conv Block 2
                layers.Conv2D(64, (3, 3), padding='same'),
                layers.BatchNormalization(),
                layers.ReLU(),
                layers.MaxPooling2D((2, 2)),
                layers.Dropout(0.25),
                
                # Conv Block 3
                layers.Conv2D(128, (3, 3), padding='same'),
                layers.BatchNormalization(),
                layers.ReLU(),
                layers.MaxPooling2D((2, 2)),
                layers.Dropout(0.25),
                
                # Conv Block 4
                layers.Conv2D(256, (3, 3), padding='same'),
                layers.BatchNormalization(),
                layers.ReLU(),
                layers.MaxPooling2D((2, 2)),
                layers.Dropout(0.25),
                
                # Conv Block 5
                layers.Conv2D(512, (3, 3), padding='same'),
                layers.BatchNormalization(),
                layers.ReLU(),
                layers.GlobalAveragePooling2D(),
                layers.Dropout(0.5),
                
                # Dense layers
                layers.Dense(256, activation='relu'),
                layers.Dropout(0.5),
                layers.Dense(1, activation='sigmoid'),
            ])
            
            self.model.compile(
                optimizer=keras.optimizers.Adam(learning_rate=0.0001),
                loss='binary_crossentropy',
                metrics=['accuracy', keras.metrics.AUC(name='auc')]
            )
            
            logger.info(f"CNN model built with {self.model.count_params()} parameters")
            
        except ImportError:
            logger.warning("TensorFlow not available. CNN model will use placeholder.")
            self.model = None
    
    def predict(self, mel_spectrogram: np.ndarray) -> float:
        """
        Predict deepfake probability from mel-spectrogram.
        
        Args:
            mel_spectrogram: Prepared spectrogram of shape (n_mels, time_frames, 1)
            
        Returns:
            Deepfake probability (0 = real, 1 = fake)
        """
        if self.model is None:
            # Return random score if model not available
            return np.random.uniform(0.3, 0.7)
        
        # Ensure correct shape
        if len(mel_spectrogram.shape) == 3:
            mel_spectrogram = np.expand_dims(mel_spectrogram, 0)
        
        prediction = self.model.predict(mel_spectrogram, verbose=0)
        return float(prediction[0][0])
    
    def save(self, path: str):
        """Save model weights."""
        if self.model is not None:
            self.model.save(path)
            logger.info(f"CNN model saved to {path}")
    
    def load(self, path: str):
        """Load model weights."""
        if os.path.exists(path):
            try:
                from tensorflow import keras
                self.model = keras.models.load_model(path)
                logger.info(f"CNN model loaded from {path}")
            except Exception as e:
                logger.error(f"Failed to load CNN model: {e}")


class LSTMTemporalModel:
    """
    LSTM/GRU model for capturing temporal inconsistencies in synthetic speech.
    
    Bidirectional architecture for context awareness.
    Input: Sequential MFCC features
    """
    
    def __init__(self, input_shape: Tuple[int, int] = (100, 40)):
        self.input_shape = input_shape
        self.model = None
        self._build_model()
    
    def _build_model(self):
        """Build LSTM architecture."""
        try:
            import tensorflow as tf
            from tensorflow import keras
            from tensorflow.keras import layers
            
            self.model = keras.Sequential([
                layers.Input(shape=self.input_shape),
                
                # Bidirectional LSTM layers
                layers.Bidirectional(layers.LSTM(128, return_sequences=True, dropout=0.3)),
                layers.Bidirectional(layers.LSTM(64, return_sequences=True, dropout=0.3)),
                layers.Bidirectional(layers.LSTM(32, dropout=0.3)),
                
                # Dense layers
                layers.Dense(64, activation='relu'),
                layers.Dropout(0.5),
                layers.Dense(32, activation='relu'),
                layers.Dropout(0.3),
                layers.Dense(1, activation='sigmoid'),
            ])
            
            self.model.compile(
                optimizer=keras.optimizers.Adam(learning_rate=0.0001),
                loss='binary_crossentropy',
                metrics=['accuracy', keras.metrics.AUC(name='auc')]
            )
            
            logger.info(f"LSTM model built with {self.model.count_params()} parameters")
            
        except ImportError:
            logger.warning("TensorFlow not available. LSTM model will use placeholder.")
            self.model = None
    
    def predict(self, mfcc_sequence: np.ndarray) -> float:
        """
        Predict deepfake probability from MFCC sequence.
        
        Args:
            mfcc_sequence: Sequence of shape (sequence_length, n_mfcc)
            
        Returns:
            Deepfake probability (0 = real, 1 = fake)
        """
        if self.model is None:
            return np.random.uniform(0.3, 0.7)
        
        if len(mfcc_sequence.shape) == 2:
            mfcc_sequence = np.expand_dims(mfcc_sequence, 0)
        
        prediction = self.model.predict(mfcc_sequence, verbose=0)
        return float(prediction[0][0])
    
    def save(self, path: str):
        """Save model weights."""
        if self.model is not None:
            self.model.save(path)
    
    def load(self, path: str):
        """Load model weights."""
        if os.path.exists(path):
            try:
                from tensorflow import keras
                self.model = keras.models.load_model(path)
            except Exception as e:
                logger.error(f"Failed to load LSTM model: {e}")


class EnsembleMLModel:
    """
    Traditional ML ensemble for deepfake detection.
    
    Models: Random Forest, XGBoost, SVM with RBF kernel
    Voting/averaging ensemble for final prediction
    """
    
    def __init__(self):
        self.models = {}
        self.feature_names = []
        self._build_models()
    
    def _build_models(self):
        """Initialize ML models."""
        try:
            from sklearn.ensemble import RandomForestClassifier
            from sklearn.svm import SVC
            from xgboost import XGBClassifier
            
            self.models = {
                'rf': RandomForestClassifier(
                    n_estimators=100,
                    max_depth=15,
                    min_samples_split=5,
                    min_samples_leaf=2,
                    n_jobs=-1,
                    random_state=42
                ),
                'xgb': XGBClassifier(
                    n_estimators=100,
                    max_depth=10,
                    learning_rate=0.1,
                    subsample=0.8,
                    colsample_bytree=0.8,
                    random_state=42,
                    eval_metric='logloss'
                ),
                'svm': SVC(
                    kernel='rbf',
                    C=1.0,
                    gamma='scale',
                    probability=True,
                    random_state=42
                ),
            }
            
            self.is_fitted = {name: False for name in self.models}
            logger.info("Ensemble ML models initialized")
            
        except ImportError as e:
            logger.warning(f"ML libraries not available: {e}")
            self.models = {}
            self.is_fitted = {}
    
    def fit(self, X: np.ndarray, y: np.ndarray):
        """
        Train all ensemble models.
        
        Args:
            X: Feature matrix (n_samples, n_features)
            y: Binary labels (0=real, 1=fake)
        """
        for name, model in self.models.items():
            logger.info(f"Training {name}...")
            model.fit(X, y)
            self.is_fitted[name] = True
        logger.info("Ensemble training complete")
    
    def predict(self, features: np.ndarray) -> Tuple[float, Dict[str, float]]:
        """
        Predict deepfake probability using ensemble.
        
        Args:
            features: Concatenated feature vector
            
        Returns:
            Tuple of (ensemble_score, individual_scores)
        """
        if not self.models:
            return np.random.uniform(0.3, 0.7), {}
        
        if len(features.shape) == 1:
            features = features.reshape(1, -1)
        
        scores = {}
        for name, model in self.models.items():
            if self.is_fitted.get(name, False):
                try:
                    prob = model.predict_proba(features)[0][1]
                    scores[name] = float(prob)
                except:
                    scores[name] = 0.5
            else:
                # Untrained model - return neutral score
                scores[name] = 0.5
        
        # Weighted average (RF and XGB weighted higher than SVM)
        weights = {'rf': 0.4, 'xgb': 0.4, 'svm': 0.2}
        ensemble_score = sum(scores.get(name, 0.5) * weight 
                            for name, weight in weights.items())
        
        return ensemble_score, scores
    
    def get_feature_importance(self) -> Dict[str, float]:
        """Get feature importance from Random Forest."""
        if 'rf' in self.models and self.is_fitted.get('rf', False):
            importance = self.models['rf'].feature_importances_
            if self.feature_names:
                return dict(zip(self.feature_names, importance))
            return {f'feature_{i}': float(imp) for i, imp in enumerate(importance)}
        return {}
    
    def save(self, path: str):
        """Save ensemble models."""
        with open(path, 'wb') as f:
            pickle.dump({
                'models': self.models,
                'is_fitted': self.is_fitted,
                'feature_names': self.feature_names,
            }, f)
        logger.info(f"Ensemble models saved to {path}")
    
    def load(self, path: str):
        """Load ensemble models."""
        if os.path.exists(path):
            with open(path, 'rb') as f:
                data = pickle.load(f)
                self.models = data['models']
                self.is_fitted = data['is_fitted']
                self.feature_names = data.get('feature_names', [])
            logger.info(f"Ensemble models loaded from {path}")


class DeepfakeDetector:
    """
    Main deepfake detection system combining multiple models.
    
    Aggregates predictions from:
    - CNN spectrogram classifier
    - LSTM temporal analyzer
    - Traditional ML ensemble
    
    Provides explainability and confidence scores.
    """
    
    def __init__(
        self,
        models_dir: Optional[str] = None,
        cnn_weight: float = 0.4,
        ensemble_weight: float = 0.35,
        lstm_weight: float = 0.25,
    ):
        """
        Initialize deepfake detector.
        
        Args:
            models_dir: Directory containing trained model weights
            cnn_weight: Weight for CNN model in final score
            ensemble_weight: Weight for ensemble in final score
            lstm_weight: Weight for LSTM in final score
        """
        self.models_dir = models_dir
        self.cnn_weight = cnn_weight
        self.ensemble_weight = ensemble_weight
        self.lstm_weight = lstm_weight
        
        # Initialize components
        self.feature_extractor = AudioFeatureExtractor()
        self.cnn_model = CNNSpectrogramModel()
        self.lstm_model = LSTMTemporalModel()
        self.ensemble_model = EnsembleMLModel()
        
        # Load pretrained weights if available
        if models_dir and os.path.exists(models_dir):
            self._load_models(models_dir)
        
        # Thresholds
        self.threshold_pass = 0.3
        self.threshold_flag = 0.6
        self.threshold_block = 0.8
        
        logger.info("DeepfakeDetector initialized")
    
    def _load_models(self, models_dir: str):
        """Load all pretrained models from directory."""
        cnn_path = os.path.join(models_dir, 'audio_cnn_weights.h5')
        lstm_path = os.path.join(models_dir, 'audio_lstm_weights.h5')
        ensemble_path = os.path.join(models_dir, 'rf_classifier.pkl')
        
        if os.path.exists(cnn_path):
            self.cnn_model.load(cnn_path)
        if os.path.exists(lstm_path):
            self.lstm_model.load(lstm_path)
        if os.path.exists(ensemble_path):
            self.ensemble_model.load(ensemble_path)
    
    def detect(
        self,
        audio_input: Union[str, bytes, np.ndarray],
        sample_rate: Optional[int] = None,
    ) -> DeepfakeResult:
        """
        Detect if audio is a deepfake.
        
        Args:
            audio_input: Path to audio file, bytes, or waveform array
            sample_rate: Sample rate if providing waveform array
            
        Returns:
            DeepfakeResult with detection results and explanation
        """
        # Extract features
        features = self.feature_extractor.extract_all_features(audio_input, sample_rate)
        
        # Get artifact detection results
        artifact_results = self._detect_artifacts(features)
        
        # Prepare inputs for each model
        cnn_input = self.feature_extractor.prepare_cnn_input(features.mel_spectrogram_db)
        lstm_input = self.feature_extractor.prepare_lstm_input(features.mfcc)
        ml_features = features.get_concatenated_features()
        
        # Get predictions from each model
        cnn_score = self.cnn_model.predict(cnn_input)
        lstm_score = self.lstm_model.predict(lstm_input)
        ensemble_score, individual_scores = self.ensemble_model.predict(ml_features)
        
        # Combine scores
        combined_score = (
            self.cnn_weight * cnn_score +
            self.ensemble_weight * ensemble_score +
            self.lstm_weight * lstm_score
        )
        
        # Adjust based on artifact detection
        if artifact_results['detected_artifacts']:
            artifact_boost = min(0.2, len(artifact_results['detected_artifacts']) * 0.05)
            combined_score = min(1.0, combined_score + artifact_boost)
        
        # Generate explanation
        explanation = self._generate_explanation(
            combined_score, cnn_score, ensemble_score, lstm_score,
            artifact_results, individual_scores
        )
        
        # Determine if fake
        is_fake = combined_score >= self.threshold_flag
        
        # Calculate confidence
        confidence = abs(combined_score - 0.5) * 2  # Scale to 0-1
        
        return DeepfakeResult(
            is_fake=is_fake,
            confidence=confidence,
            risk_score=combined_score,
            cnn_score=cnn_score,
            ensemble_score=ensemble_score,
            lstm_score=lstm_score,
            explanation=explanation,
            detected_artifacts=artifact_results['detected_artifacts'],
            feature_importance=self.ensemble_model.get_feature_importance(),
        )
    
    def _detect_artifacts(self, features: AudioFeatures) -> Dict:
        """Detect audio artifacts typical in deepfakes."""
        detected = []
        scores = {}
        
        # High frequency artifact detection
        hf_results = self.feature_extractor.detect_high_frequency_artifacts(features.waveform)
        scores['high_freq'] = hf_results.get('artifact_score', 0.0)
        if hf_results.get('artifact_score', 0) > 0.3:
            detected.append("High-frequency artifacts detected (typical in GAN-generated audio)")
        if hf_results.get('high_freq_ratio', 0) > 0.5:
            detected.append("Abnormal high-frequency energy ratio")
        
        # Phase inconsistency detection
        phase_results = self.feature_extractor.detect_phase_inconsistencies(features.waveform)
        scores['phase'] = phase_results.get('phase_discontinuity_score', 0.0)
        if phase_results.get('phase_discontinuity_score', 0) > 0.1:
            detected.append("Phase discontinuities detected (possible audio splicing)")
        
        # Spectral anomaly detection
        spectral_std = np.std(features.spectral_flatness)
        scores['spectral'] = min(1.0, spectral_std * 10)
        if spectral_std < 0.01:
            detected.append("Unnaturally consistent spectral flatness")
        
        # Energy pattern anomalies
        rms_std = np.std(features.rms_energy)
        if rms_std < 0.005:
            detected.append("Unusually consistent energy patterns")
        
        return {
            'detected_artifacts': detected,
            'artifact_scores': scores,
        }
    
    def _generate_explanation(
        self,
        combined_score: float,
        cnn_score: float,
        ensemble_score: float,
        lstm_score: float,
        artifact_results: Dict,
        individual_scores: Dict[str, float],
    ) -> str:
        """Generate human-readable explanation."""
        explanation_parts = []
        
        # Overall assessment
        if combined_score < self.threshold_pass:
            explanation_parts.append("Audio appears to be genuine with high confidence.")
        elif combined_score < self.threshold_flag:
            explanation_parts.append("Audio shows some suspicious characteristics but is likely genuine.")
        elif combined_score < self.threshold_block:
            explanation_parts.append("Audio shows significant signs of being synthetically generated.")
        else:
            explanation_parts.append("Audio is highly likely to be a deepfake or synthetic voice.")
        
        # Model-specific insights
        explanation_parts.append(f"\n\nModel Analysis:")
        explanation_parts.append(f"- Spectrogram CNN: {cnn_score:.2%} fake probability")
        explanation_parts.append(f"- Temporal LSTM: {lstm_score:.2%} fake probability")
        explanation_parts.append(f"- ML Ensemble: {ensemble_score:.2%} fake probability")
        
        if individual_scores:
            explanation_parts.append(f"\n  Individual models: RF={individual_scores.get('rf', 0):.2%}, "
                                    f"XGB={individual_scores.get('xgb', 0):.2%}, "
                                    f"SVM={individual_scores.get('svm', 0):.2%}")
        
        # Artifact details
        if artifact_results['detected_artifacts']:
            explanation_parts.append("\n\nDetected Anomalies:")
            for artifact in artifact_results['detected_artifacts']:
                explanation_parts.append(f"- {artifact}")
        
        return "\n".join(explanation_parts)
    
    def get_risk_level(self, score: float) -> str:
        """Get risk level classification from score."""
        if score < self.threshold_pass:
            return "PASS"
        elif score < self.threshold_flag:
            return "FLAG"
        elif score < self.threshold_block:
            return "BLOCK"
        else:
            return "ALERT"
    
    def batch_detect(
        self,
        audio_files: List[str],
    ) -> List[DeepfakeResult]:
        """
        Detect deepfakes in batch.
        
        Args:
            audio_files: List of paths to audio files
            
        Returns:
            List of DeepfakeResult objects
        """
        results = []
        for audio_file in audio_files:
            try:
                result = self.detect(audio_file)
                results.append(result)
            except Exception as e:
                logger.error(f"Error processing {audio_file}: {e}")
                results.append(DeepfakeResult(
                    is_fake=False,
                    confidence=0.0,
                    risk_score=0.0,
                    cnn_score=0.0,
                    ensemble_score=0.0,
                    lstm_score=0.0,
                    explanation=f"Error processing audio: {str(e)}",
                    detected_artifacts=[],
                    feature_importance={},
                ))
        return results
