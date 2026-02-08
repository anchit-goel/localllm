"""
Voice Cloning Detection Module.

Detects attempts to use cloned voices through:
- Speaker embedding extraction and comparison
- Cosine similarity against known voice database
- Replay attack detection via phase analysis
- High-frequency artifact detection
"""

import numpy as np
from typing import Dict, Optional, List, Tuple, Union
from dataclasses import dataclass
import logging
import os
import pickle

from .feature_extraction import AudioFeatureExtractor

logger = logging.getLogger(__name__)


@dataclass
class VoiceCloneResult:
    """Container for voice clone detection results."""
    is_cloned: bool
    similarity_score: float
    matched_speaker: Optional[str]
    confidence: float
    replay_attack_detected: bool
    embedding: np.ndarray
    explanation: str
    artifact_indicators: List[str]
    
    def to_dict(self) -> Dict:
        """Convert to dictionary for API response."""
        return {
            'is_cloned': self.is_cloned,
            'similarity_score': self.similarity_score,
            'matched_speaker': self.matched_speaker,
            'confidence': self.confidence,
            'replay_attack_detected': self.replay_attack_detected,
            'explanation': self.explanation,
            'artifact_indicators': self.artifact_indicators,
        }


class SpeakerEmbedder:
    """
    Extract speaker embeddings for voice comparison.
    
    Uses either resemblyzer (pretrained) or custom embedding model.
    """
    
    def __init__(self, use_pretrained: bool = True):
        self.use_pretrained = use_pretrained
        self.encoder = None
        self._initialize()
    
    def _initialize(self):
        """Initialize speaker encoder."""
        if self.use_pretrained:
            try:
                from resemblyzer import VoiceEncoder
                self.encoder = VoiceEncoder()
                logger.info("Resemblyzer voice encoder loaded")
            except ImportError:
                logger.warning("Resemblyzer not available. Using fallback embedding.")
                self.encoder = None
    
    def extract_embedding(
        self,
        waveform: np.ndarray,
        sample_rate: int = 16000,
    ) -> np.ndarray:
        """
        Extract speaker embedding from audio.
        
        Args:
            waveform: Audio waveform
            sample_rate: Sample rate
            
        Returns:
            Speaker embedding vector (256-dim for resemblyzer)
        """
        if self.encoder is not None:
            try:
                from resemblyzer import preprocess_wav
                # Preprocess for resemblyzer
                wav = preprocess_wav(waveform, sample_rate)
                embedding = self.encoder.embed_utterance(wav)
                return embedding
            except Exception as e:
                logger.error(f"Error extracting embedding: {e}")
        
        # Fallback: Use MFCC-based pseudo-embedding
        return self._fallback_embedding(waveform, sample_rate)
    
    def _fallback_embedding(
        self,
        waveform: np.ndarray,
        sample_rate: int,
    ) -> np.ndarray:
        """Generate pseudo-embedding from MFCC statistics."""
        try:
            import librosa
            
            # Extract MFCCs
            mfcc = librosa.feature.mfcc(y=waveform, sr=sample_rate, n_mfcc=40)
            
            # Create embedding from MFCC statistics
            embedding = np.concatenate([
                np.mean(mfcc, axis=1),
                np.std(mfcc, axis=1),
                np.min(mfcc, axis=1),
                np.max(mfcc, axis=1),
                np.median(mfcc, axis=1),
                np.percentile(mfcc, 25, axis=1),
                np.percentile(mfcc, 75, axis=1),
            ])
            
            # Normalize
            embedding = embedding / (np.linalg.norm(embedding) + 1e-10)
            
            return embedding
            
        except Exception as e:
            logger.error(f"Error in fallback embedding: {e}")
            return np.random.randn(256)  # Random embedding as last resort


class VoiceDatabase:
    """
    Database of known speaker embeddings for comparison.
    """
    
    def __init__(self, db_path: Optional[str] = None):
        self.embeddings: Dict[str, List[np.ndarray]] = {}
        self.metadata: Dict[str, Dict] = {}
        self.db_path = db_path
        
        if db_path and os.path.exists(db_path):
            self.load(db_path)
    
    def add_speaker(
        self,
        speaker_id: str,
        embedding: np.ndarray,
        metadata: Optional[Dict] = None,
    ):
        """Add speaker embedding to database."""
        if speaker_id not in self.embeddings:
            self.embeddings[speaker_id] = []
        self.embeddings[speaker_id].append(embedding)
        
        if metadata:
            self.metadata[speaker_id] = metadata
    
    def get_speaker_centroid(self, speaker_id: str) -> np.ndarray:
        """Get average embedding for speaker."""
        if speaker_id not in self.embeddings:
            raise ValueError(f"Speaker {speaker_id} not in database")
        
        embeddings = np.array(self.embeddings[speaker_id])
        return np.mean(embeddings, axis=0)
    
    def find_closest_speaker(
        self,
        embedding: np.ndarray,
        threshold: float = 0.7,
    ) -> Tuple[Optional[str], float]:
        """
        Find closest speaker to given embedding.
        
        Args:
            embedding: Query embedding
            threshold: Minimum similarity to consider a match
            
        Returns:
            Tuple of (speaker_id or None, similarity_score)
        """
        best_speaker = None
        best_score = 0.0
        
        for speaker_id in self.embeddings:
            centroid = self.get_speaker_centroid(speaker_id)
            similarity = self._cosine_similarity(embedding, centroid)
            
            if similarity > best_score:
                best_score = similarity
                if similarity >= threshold:
                    best_speaker = speaker_id
        
        return best_speaker, best_score
    
    def _cosine_similarity(self, a: np.ndarray, b: np.ndarray) -> float:
        """Compute cosine similarity between two vectors."""
        return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-10))
    
    def save(self, path: str):
        """Save database to file."""
        with open(path, 'wb') as f:
            pickle.dump({
                'embeddings': self.embeddings,
                'metadata': self.metadata,
            }, f)
        logger.info(f"Voice database saved to {path}")
    
    def load(self, path: str):
        """Load database from file."""
        with open(path, 'rb') as f:
            data = pickle.load(f)
            self.embeddings = data['embeddings']
            self.metadata = data.get('metadata', {})
        logger.info(f"Voice database loaded from {path}")


class ReplayAttackDetector:
    """
    Detect replay attacks through audio analysis.
    
    Analyzes:
    - Phase characteristics
    - High-frequency artifacts
    - Recording environment signatures
    """
    
    def __init__(self):
        self.feature_extractor = AudioFeatureExtractor()
    
    def detect(
        self,
        waveform: np.ndarray,
        sample_rate: int = 16000,
    ) -> Dict:
        """
        Detect replay attack indicators.
        
        Args:
            waveform: Audio waveform
            sample_rate: Sample rate
            
        Returns:
            Dictionary with detection results
        """
        results = {
            'is_replay': False,
            'confidence': 0.0,
            'indicators': [],
            'scores': {},
        }
        
        # Phase analysis
        phase_results = self.feature_extractor.detect_phase_inconsistencies(waveform)
        results['scores']['phase'] = phase_results.get('phase_discontinuity_score', 0)
        
        if phase_results.get('phase_discontinuity_score', 0) > 0.15:
            results['indicators'].append("Phase discontinuities suggest replay or splicing")
        
        # High-frequency artifact analysis
        hf_results = self.feature_extractor.detect_high_frequency_artifacts(waveform)
        results['scores']['high_freq'] = hf_results.get('artifact_score', 0)
        
        if hf_results.get('high_freq_ratio', 0) < 0.01:
            results['indicators'].append("Suspiciously low high-frequency content (possible bandlimited replay)")
        
        if hf_results.get('artifact_score', 0) > 0.4:
            results['indicators'].append("High-frequency artifacts detected")
        
        # Room acoustics detection (simple version)
        room_score = self._detect_room_artifacts(waveform, sample_rate)
        results['scores']['room'] = room_score
        
        if room_score > 0.6:
            results['indicators'].append("Double room impulse response detected (speaker playback)")
        
        # Background noise analysis
        noise_score = self._analyze_background_noise(waveform, sample_rate)
        results['scores']['noise'] = noise_score
        
        if noise_score > 0.5:
            results['indicators'].append("Unusual background noise pattern")
        
        # Calculate overall replay probability
        overall_score = np.mean([
            phase_results.get('phase_discontinuity_score', 0) * 2,
            hf_results.get('artifact_score', 0),
            room_score,
            noise_score,
        ])
        
        results['is_replay'] = overall_score > 0.4
        results['confidence'] = min(1.0, overall_score)
        
        return results
    
    def _detect_room_artifacts(self, waveform: np.ndarray, sample_rate: int) -> float:
        """Detect room impulse response artifacts."""
        try:
            from scipy.signal import correlate
            
            # Compute autocorrelation
            autocorr = correlate(waveform, waveform, mode='same')
            autocorr = autocorr / np.max(autocorr)
            
            # Look for secondary peaks (room reflections)
            center = len(autocorr) // 2
            search_range = int(sample_rate * 0.1)  # 100ms
            
            secondary_region = autocorr[center + 100:center + search_range]
            if len(secondary_region) > 0:
                peak_ratio = np.max(secondary_region)
                return min(1.0, peak_ratio * 2)
            
        except Exception as e:
            logger.error(f"Error in room artifact detection: {e}")
        
        return 0.0
    
    def _analyze_background_noise(self, waveform: np.ndarray, sample_rate: int) -> float:
        """Analyze background noise for replay indicators."""
        try:
            # Use low-energy segments to estimate noise
            import librosa
            
            rms = librosa.feature.rms(y=waveform)[0]
            noise_threshold = np.percentile(rms, 10)
            
            # Check for unusual noise patterns
            noise_variance = np.var(rms[rms < noise_threshold * 2])
            
            # High variance in noise might indicate replay
            return min(1.0, noise_variance * 100)
            
        except Exception as e:
            logger.error(f"Error in noise analysis: {e}")
        
        return 0.0


class VoiceCloneDetector:
    """
    Main voice cloning detection system.
    
    Combines:
    - Speaker embedding comparison
    - Replay attack detection
    - Synthetic voice artifact detection
    """
    
    def __init__(
        self,
        voice_db_path: Optional[str] = None,
        similarity_threshold: float = 0.75,
    ):
        """
        Initialize voice clone detector.
        
        Args:
            voice_db_path: Path to voice embeddings database
            similarity_threshold: Threshold for speaker matching
        """
        self.similarity_threshold = similarity_threshold
        
        # Initialize components
        self.feature_extractor = AudioFeatureExtractor()
        self.embedder = SpeakerEmbedder()
        self.voice_db = VoiceDatabase(voice_db_path)
        self.replay_detector = ReplayAttackDetector()
        
        logger.info("VoiceCloneDetector initialized")
    
    def register_speaker(
        self,
        speaker_id: str,
        audio_samples: List[Union[str, np.ndarray]],
        metadata: Optional[Dict] = None,
    ):
        """
        Register a speaker in the database.
        
        Args:
            speaker_id: Unique speaker identifier
            audio_samples: List of audio file paths or waveforms
            metadata: Optional speaker metadata
        """
        for sample in audio_samples:
            if isinstance(sample, str):
                waveform, sr = self.feature_extractor.load_audio(sample)
            else:
                waveform = sample
                sr = self.feature_extractor.sample_rate
            
            embedding = self.embedder.extract_embedding(waveform, sr)
            self.voice_db.add_speaker(speaker_id, embedding, metadata)
        
        logger.info(f"Registered speaker {speaker_id} with {len(audio_samples)} samples")
    
    def verify_speaker(
        self,
        audio_input: Union[str, bytes, np.ndarray],
        claimed_speaker: str,
        sample_rate: Optional[int] = None,
    ) -> VoiceCloneResult:
        """
        Verify if audio matches claimed speaker.
        
        Args:
            audio_input: Audio to verify
            claimed_speaker: Expected speaker ID
            sample_rate: Sample rate if providing waveform
            
        Returns:
            VoiceCloneResult with verification results
        """
        # Load audio
        if isinstance(audio_input, str):
            waveform, sr = self.feature_extractor.load_audio(audio_input)
        elif isinstance(audio_input, bytes):
            waveform, sr = self.feature_extractor.load_audio_from_bytes(audio_input)
        else:
            waveform = audio_input
            sr = sample_rate or self.feature_extractor.sample_rate
        
        # Extract embedding
        embedding = self.embedder.extract_embedding(waveform, sr)
        
        # Compare with claimed speaker
        if claimed_speaker in self.voice_db.embeddings:
            centroid = self.voice_db.get_speaker_centroid(claimed_speaker)
            similarity = float(np.dot(embedding, centroid) / 
                             (np.linalg.norm(embedding) * np.linalg.norm(centroid) + 1e-10))
        else:
            similarity = 0.0
        
        # Check for replay attack
        replay_result = self.replay_detector.detect(waveform, sr)
        
        # Detect cloning artifacts
        artifact_indicators = self._detect_clone_artifacts(waveform, sr)
        artifact_indicators.extend(replay_result['indicators'])
        
        # Determine if cloned
        is_cloned = (
            similarity < self.similarity_threshold or
            replay_result['is_replay'] or
            len(artifact_indicators) >= 2
        )
        
        # Calculate confidence
        confidence = self._calculate_confidence(
            similarity, replay_result['confidence'], len(artifact_indicators)
        )
        
        # Generate explanation
        explanation = self._generate_explanation(
            similarity, claimed_speaker, replay_result, artifact_indicators, is_cloned
        )
        
        return VoiceCloneResult(
            is_cloned=is_cloned,
            similarity_score=similarity,
            matched_speaker=claimed_speaker if similarity >= self.similarity_threshold else None,
            confidence=confidence,
            replay_attack_detected=replay_result['is_replay'],
            embedding=embedding,
            explanation=explanation,
            artifact_indicators=artifact_indicators,
        )
    
    def detect(
        self,
        audio_input: Union[str, bytes, np.ndarray],
        sample_rate: Optional[int] = None,
    ) -> VoiceCloneResult:
        """
        Detect voice cloning without specific speaker claim.
        
        Args:
            audio_input: Audio to analyze
            sample_rate: Sample rate if providing waveform
            
        Returns:
            VoiceCloneResult with detection results
        """
        # Load audio
        if isinstance(audio_input, str):
            waveform, sr = self.feature_extractor.load_audio(audio_input)
        elif isinstance(audio_input, bytes):
            waveform, sr = self.feature_extractor.load_audio_from_bytes(audio_input)
        else:
            waveform = audio_input
            sr = sample_rate or self.feature_extractor.sample_rate
        
        # Extract embedding
        embedding = self.embedder.extract_embedding(waveform, sr)
        
        # Find closest speaker in database
        matched_speaker, similarity = self.voice_db.find_closest_speaker(
            embedding, self.similarity_threshold
        )
        
        # Check for replay attack
        replay_result = self.replay_detector.detect(waveform, sr)
        
        # Detect cloning artifacts
        artifact_indicators = self._detect_clone_artifacts(waveform, sr)
        artifact_indicators.extend(replay_result['indicators'])
        
        # Determine if cloned (more aggressive without claimed speaker)
        is_cloned = (
            replay_result['is_replay'] or
            len(artifact_indicators) >= 2
        )
        
        confidence = self._calculate_confidence(
            similarity, replay_result['confidence'], len(artifact_indicators)
        )
        
        explanation = self._generate_explanation(
            similarity, matched_speaker, replay_result, artifact_indicators, is_cloned
        )
        
        return VoiceCloneResult(
            is_cloned=is_cloned,
            similarity_score=similarity,
            matched_speaker=matched_speaker,
            confidence=confidence,
            replay_attack_detected=replay_result['is_replay'],
            embedding=embedding,
            explanation=explanation,
            artifact_indicators=artifact_indicators,
        )
    
    def _detect_clone_artifacts(self, waveform: np.ndarray, sample_rate: int) -> List[str]:
        """Detect artifacts typical in cloned voices."""
        indicators = []
        
        # High-frequency analysis
        hf_results = self.feature_extractor.detect_high_frequency_artifacts(waveform)
        if hf_results.get('artifact_score', 0) > 0.35:
            indicators.append("Synthetic voice high-frequency artifacts")
        
        # Phase analysis
        phase_results = self.feature_extractor.detect_phase_inconsistencies(waveform)
        if phase_results.get('phase_discontinuity_score', 0) > 0.1:
            indicators.append("Phase inconsistencies suggesting voice synthesis")
        
        # Spectral consistency (cloned voices often have unnatural consistency)
        features = self.feature_extractor.extract_all_features(waveform, sample_rate)
        spectral_std = np.std(features.spectral_centroid)
        if spectral_std < 50:
            indicators.append("Unnaturally consistent spectral characteristics")
        
        # Pitch stability (clone voices often too stable)
        try:
            import librosa
            pitches, magnitudes = librosa.piptrack(y=waveform, sr=sample_rate)
            valid_pitches = pitches[magnitudes > np.median(magnitudes)]
            if len(valid_pitches) > 0:
                pitch_std = np.std(valid_pitches)
                if pitch_std < 10:
                    indicators.append("Unusually stable pitch (possible TTS)")
        except:
            pass
        
        return indicators
    
    def _calculate_confidence(
        self,
        similarity: float,
        replay_confidence: float,
        num_artifacts: int,
    ) -> float:
        """Calculate overall detection confidence."""
        # Higher artifacts and replay confidence increase clone detection confidence
        artifact_conf = min(1.0, num_artifacts * 0.2)
        
        # Low similarity increases clone confidence
        similarity_conf = max(0, 1 - similarity)
        
        return min(1.0, (replay_confidence + artifact_conf + similarity_conf) / 2)
    
    def _generate_explanation(
        self,
        similarity: float,
        speaker: Optional[str],
        replay_result: Dict,
        artifact_indicators: List[str],
        is_cloned: bool,
    ) -> str:
        """Generate human-readable explanation."""
        parts = []
        
        if is_cloned:
            parts.append("⚠️ Voice cloning or impersonation detected!")
        else:
            parts.append("✓ Voice appears to be genuine.")
        
        parts.append(f"\n\nSpeaker Similarity: {similarity:.2%}")
        if speaker:
            parts.append(f"Matched Speaker: {speaker}")
        else:
            parts.append("No matching speaker found in database")
        
        if replay_result['is_replay']:
            parts.append("\n\n🔴 REPLAY ATTACK DETECTED")
            parts.append(f"Confidence: {replay_result['confidence']:.2%}")
        
        if artifact_indicators:
            parts.append("\n\nDetected Indicators:")
            for indicator in artifact_indicators:
                parts.append(f"  • {indicator}")
        
        return "\n".join(parts)
    
    def save_database(self, path: str):
        """Save voice database."""
        self.voice_db.save(path)
    
    def load_database(self, path: str):
        """Load voice database."""
        self.voice_db.load(path)
