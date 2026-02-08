"""
Cross-Modal Consistency Checker.

Verifies consistency between audio and visual content:
- Text extraction from both modalities
- Semantic alignment scoring
- Contradiction detection
- CLIP-based similarity (optional)
"""

import numpy as np
from typing import Dict, Optional, List, Union
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)


@dataclass
class ConsistencyResult:
    """Container for consistency check results."""
    is_consistent: bool
    consistency_score: float
    audio_content: str
    visual_content: str
    mismatches: List[str]
    semantic_similarity: float
    clip_similarity: Optional[float]
    explanation: str
    
    def to_dict(self) -> Dict:
        return {
            'is_consistent': self.is_consistent,
            'consistency_score': self.consistency_score,
            'audio_content': self.audio_content,
            'visual_content': self.visual_content,
            'mismatches': self.mismatches,
            'semantic_similarity': self.semantic_similarity,
            'clip_similarity': self.clip_similarity,
            'explanation': self.explanation,
        }


class ConsistencyChecker:
    """
    Cross-modal consistency verification.
    
    Compares audio and visual content for:
    - Semantic alignment
    - Contradiction detection
    - Content mismatch identification
    """
    
    def __init__(self, use_clip: bool = False):
        """
        Initialize consistency checker.
        
        Args:
            use_clip: Whether to use CLIP for image-text similarity
        """
        self.use_clip = use_clip
        self.clip_model = None
        self.clip_processor = None
        
        if use_clip:
            self._load_clip()
        
        # Contradiction patterns
        self.contradiction_pairs = [
            ('yes', 'no'), ('true', 'false'), ('confirm', 'deny'),
            ('accept', 'reject'), ('allow', 'block'), ('safe', 'dangerous'),
            ('real', 'fake'), ('genuine', 'forged'), ('authentic', 'fraudulent'),
        ]
        
        logger.info("ConsistencyChecker initialized")
    
    def _load_clip(self):
        """Load CLIP model for image-text similarity."""
        try:
            from transformers import CLIPProcessor, CLIPModel
            self.clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
            self.clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
            logger.info("CLIP model loaded")
        except Exception as e:
            logger.warning(f"Could not load CLIP: {e}")
            self.use_clip = False
    
    def extract_audio_transcript(self, audio_input: Union[str, bytes]) -> str:
        """
        Extract transcript from audio using speech recognition.
        
        Args:
            audio_input: Path or bytes of audio file
            
        Returns:
            Transcribed text
        """
        try:
            import speech_recognition as sr
            recognizer = sr.Recognizer()
            
            if isinstance(audio_input, str):
                with sr.AudioFile(audio_input) as source:
                    audio_data = recognizer.record(source)
            else:
                import io
                import soundfile as sf
                import tempfile
                
                with io.BytesIO(audio_input) as f:
                    data, sr_rate = sf.read(f)
                
                with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp:
                    sf.write(tmp.name, data, sr_rate)
                    with sr.AudioFile(tmp.name) as source:
                        audio_data = recognizer.record(source)
            
            return recognizer.recognize_google(audio_data)
            
        except Exception as e:
            logger.warning(f"Speech recognition failed: {e}")
            return ""
    
    def calculate_semantic_similarity(self, text1: str, text2: str) -> float:
        """
        Calculate semantic similarity between two texts.
        
        Uses simple word overlap + TF-IDF when available.
        
        Args:
            text1: First text
            text2: Second text
            
        Returns:
            Similarity score (0-1)
        """
        if not text1 or not text2:
            return 0.0
        
        # Tokenize
        words1 = set(text1.lower().split())
        words2 = set(text2.lower().split())
        
        # Jaccard similarity
        intersection = words1 & words2
        union = words1 | words2
        
        if not union:
            return 0.0
        
        jaccard = len(intersection) / len(union)
        
        # Try to use more sophisticated similarity
        try:
            from sklearn.feature_extraction.text import TfidfVectorizer
            from sklearn.metrics.pairwise import cosine_similarity
            
            vectorizer = TfidfVectorizer()
            tfidf = vectorizer.fit_transform([text1, text2])
            similarity = cosine_similarity(tfidf[0:1], tfidf[1:2])[0][0]
            
            return float(similarity)
        except:
            return jaccard
    
    def calculate_clip_similarity(
        self,
        image: np.ndarray,
        text: str,
    ) -> Optional[float]:
        """
        Calculate CLIP-based image-text similarity.
        
        Args:
            image: Image as numpy array
            text: Text to compare
            
        Returns:
            CLIP similarity score or None if unavailable
        """
        if not self.use_clip or self.clip_model is None:
            return None
        
        try:
            from PIL import Image as PILImage
            import torch
            
            # Convert numpy to PIL
            pil_image = PILImage.fromarray(image)
            
            # Process inputs
            inputs = self.clip_processor(
                text=[text],
                images=pil_image,
                return_tensors="pt",
                padding=True,
            )
            
            # Get similarity
            with torch.no_grad():
                outputs = self.clip_model(**inputs)
                logits = outputs.logits_per_image
                similarity = torch.sigmoid(logits).item()
            
            return float(similarity)
            
        except Exception as e:
            logger.warning(f"CLIP similarity failed: {e}")
            return None
    
    def detect_contradictions(self, text1: str, text2: str) -> List[str]:
        """
        Detect contradictions between two texts.
        
        Args:
            text1: First text
            text2: Second text
            
        Returns:
            List of detected contradictions
        """
        contradictions = []
        text1_lower = text1.lower()
        text2_lower = text2.lower()
        
        for word1, word2 in self.contradiction_pairs:
            if word1 in text1_lower and word2 in text2_lower:
                contradictions.append(f"'{word1}' in audio vs '{word2}' in visual")
            if word2 in text1_lower and word1 in text2_lower:
                contradictions.append(f"'{word2}' in audio vs '{word1}' in visual")
        
        return contradictions
    
    def detect_topic_mismatch(self, text1: str, text2: str) -> List[str]:
        """
        Detect topic mismatches between texts.
        
        Args:
            text1: First text (audio)
            text2: Second text (visual)
            
        Returns:
            List of mismatches
        """
        mismatches = []
        
        # Extract key topics (simple noun phrase extraction)
        import re
        
        # Get capitalized words (potential proper nouns/topics)
        topics1 = set(re.findall(r'\b[A-Z][a-z]+\b', text1))
        topics2 = set(re.findall(r'\b[A-Z][a-z]+\b', text2))
        
        # Topics unique to each
        audio_only = topics1 - topics2
        visual_only = topics2 - topics1
        
        if audio_only:
            mismatches.append(f"Topics in audio only: {', '.join(list(audio_only)[:5])}")
        if visual_only:
            mismatches.append(f"Topics in visual only: {', '.join(list(visual_only)[:5])}")
        
        return mismatches
    
    def check(
        self,
        audio_content: str,
        visual_content: str,
        image: Optional[np.ndarray] = None,
    ) -> ConsistencyResult:
        """
        Check consistency between audio and visual content.
        
        Args:
            audio_content: Transcribed audio or text from audio
            visual_content: Text extracted from visual (OCR)
            image: Optional image for CLIP analysis
            
        Returns:
            ConsistencyResult with analysis
        """
        # Calculate semantic similarity
        semantic_sim = self.calculate_semantic_similarity(audio_content, visual_content)
        
        # Calculate CLIP similarity if image provided
        clip_sim = None
        if image is not None and audio_content:
            clip_sim = self.calculate_clip_similarity(image, audio_content)
        
        # Detect contradictions
        contradictions = self.detect_contradictions(audio_content, visual_content)
        
        # Detect topic mismatches
        topic_mismatches = self.detect_topic_mismatch(audio_content, visual_content)
        
        # Combine mismatches
        all_mismatches = contradictions + topic_mismatches
        
        # Calculate consistency score
        base_score = semantic_sim
        
        # Penalize contradictions heavily
        contradiction_penalty = len(contradictions) * 0.15
        mismatch_penalty = len(topic_mismatches) * 0.05
        
        consistency_score = max(0, base_score - contradiction_penalty - mismatch_penalty)
        
        # Boost with CLIP if available
        if clip_sim is not None:
            consistency_score = (consistency_score + clip_sim) / 2
        
        # Determine consistency
        is_consistent = consistency_score > 0.5 and not contradictions
        
        # Generate explanation
        explanation = self._generate_explanation(
            is_consistent, consistency_score, semantic_sim,
            clip_sim, contradictions, topic_mismatches
        )
        
        return ConsistencyResult(
            is_consistent=is_consistent,
            consistency_score=consistency_score,
            audio_content=audio_content[:500],
            visual_content=visual_content[:500],
            mismatches=all_mismatches,
            semantic_similarity=semantic_sim,
            clip_similarity=clip_sim,
            explanation=explanation,
        )
    
    def _generate_explanation(
        self,
        consistent: bool,
        score: float,
        semantic: float,
        clip: Optional[float],
        contradictions: List[str],
        mismatches: List[str],
    ) -> str:
        """Generate human-readable explanation."""
        parts = []
        
        if consistent:
            parts.append("✓ Audio and visual content are consistent")
        else:
            parts.append("⚠️ Inconsistency detected between audio and visual content")
        
        parts.append(f"\nConsistency Score: {score:.2%}")
        parts.append(f"Semantic Similarity: {semantic:.2%}")
        
        if clip is not None:
            parts.append(f"CLIP Similarity: {clip:.2%}")
        
        if contradictions:
            parts.append("\n\n🔴 Contradictions:")
            for c in contradictions:
                parts.append(f"  • {c}")
        
        if mismatches:
            parts.append("\n\nTopic Mismatches:")
            for m in mismatches:
                parts.append(f"  • {m}")
        
        return "\n".join(parts)
