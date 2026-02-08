"""
Steganography Detection Module.

Detects hidden data in images through:
- LSB (Least Significant Bit) analysis
- Color histogram anomalies
- Perceptual hashing for modifications
- Statistical analysis
"""

import numpy as np
from typing import Dict, Optional, List, Union
from dataclasses import dataclass
import logging

try:
    import cv2
except ImportError:
    cv2 = None

try:
    from PIL import Image
except ImportError:
    Image = None

logger = logging.getLogger(__name__)


@dataclass
class SteganographyResult:
    """Container for steganography detection results."""
    hidden_data_detected: bool
    confidence: float
    detection_methods: List[str]
    lsb_score: float
    histogram_score: float
    statistical_score: float
    explanation: str
    
    def to_dict(self) -> Dict:
        return {
            'hidden_data_detected': self.hidden_data_detected,
            'confidence': self.confidence,
            'detection_methods': self.detection_methods,
            'lsb_score': self.lsb_score,
            'histogram_score': self.histogram_score,
            'statistical_score': self.statistical_score,
            'explanation': self.explanation,
        }


class SteganographyChecker:
    """
    Steganography detection for hidden data in images.
    
    Implements multiple detection techniques:
    - LSB analysis for embedded data
    - Chi-square analysis
    - RS steganalysis
    - Histogram analysis
    """
    
    def __init__(self, sensitivity: float = 0.5):
        """
        Initialize steganography checker.
        
        Args:
            sensitivity: Detection sensitivity (0-1, higher = more sensitive)
        """
        self.sensitivity = sensitivity
        logger.info("SteganographyChecker initialized")
    
    def load_image(self, image_input: Union[str, bytes, np.ndarray]) -> np.ndarray:
        """Load image from various sources."""
        if cv2 is None:
            raise ImportError("OpenCV required for steganography detection")
        
        if isinstance(image_input, str):
            image = cv2.imread(image_input)
        elif isinstance(image_input, bytes):
            nparr = np.frombuffer(image_input, np.uint8)
            image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        elif isinstance(image_input, np.ndarray):
            image = image_input
        else:
            raise ValueError(f"Unsupported input type: {type(image_input)}")
        
        return image
    
    def analyze_lsb(self, image: np.ndarray) -> Dict:
        """
        Analyze LSB (Least Significant Bit) patterns.
        
        LSB steganography embeds data in the least significant bits
        of pixel values. This creates detectable statistical anomalies.
        
        Args:
            image: Image as numpy array
            
        Returns:
            Dictionary with LSB analysis results
        """
        # Extract LSB planes for each channel
        lsb_planes = []
        for i in range(3):  # BGR channels
            channel = image[:, :, i]
            lsb = channel & 1  # Extract LSB
            lsb_planes.append(lsb)
        
        results = {
            'randomness_scores': [],
            'pattern_detected': False,
            'entropy_scores': [],
        }
        
        for i, lsb in enumerate(lsb_planes):
            # Calculate randomness (should be ~0.5 for natural images)
            randomness = np.mean(lsb)
            results['randomness_scores'].append(float(randomness))
            
            # Calculate entropy
            hist = np.bincount(lsb.flatten(), minlength=2)
            probs = hist / hist.sum()
            entropy = -np.sum(probs * np.log2(probs + 1e-10))
            results['entropy_scores'].append(float(entropy))
            
            # Check for patterns (horizontal/vertical correlations)
            h_corr = np.corrcoef(lsb[:, :-1].flatten(), lsb[:, 1:].flatten())[0, 1]
            v_corr = np.corrcoef(lsb[:-1, :].flatten(), lsb[1:, :].flatten())[0, 1]
            
            # Natural images have some correlation, steganographic images less
            if abs(h_corr) < 0.1 and abs(v_corr) < 0.1:
                results['pattern_detected'] = True
        
        # Calculate overall LSB score
        avg_randomness = np.mean(results['randomness_scores'])
        avg_entropy = np.mean(results['entropy_scores'])
        
        # Perfect randomness (0.5) with high entropy suggests steganography
        randomness_deviation = abs(avg_randomness - 0.5)
        
        # Score: low deviation from 0.5 AND high entropy = suspicious
        if randomness_deviation < 0.1 and avg_entropy > 0.95:
            results['lsb_score'] = 0.8
        elif randomness_deviation < 0.15 and avg_entropy > 0.9:
            results['lsb_score'] = 0.5
        else:
            results['lsb_score'] = 0.2
        
        return results
    
    def chi_square_analysis(self, image: np.ndarray) -> Dict:
        """
        Chi-square statistical analysis for LSB steganography.
        
        Compares pairs of values (2i, 2i+1) which should be equal
        after LSB embedding.
        
        Args:
            image: Image as numpy array
            
        Returns:
            Dictionary with chi-square results
        """
        results = {'chi_square_values': [], 'p_values': []}
        
        for channel in range(3):
            pixels = image[:, :, channel].flatten()
            
            # Count pairs of values
            pairs = {}
            for val in pixels:
                pair = val - (val % 2)  # Round down to even
                pairs[pair] = pairs.get(pair, [0, 0])
                pairs[pair][val % 2] += 1
            
            # Calculate chi-square
            chi_sq = 0
            for pair, counts in pairs.items():
                expected = sum(counts) / 2
                if expected > 0:
                    chi_sq += sum((c - expected) ** 2 / expected for c in counts)
            
            results['chi_square_values'].append(float(chi_sq))
        
        # Low chi-square indicates LSB steganography
        avg_chi_sq = np.mean(results['chi_square_values'])
        
        # Normalize to 0-1 score (lower chi-sq = higher score)
        results['chi_score'] = max(0, 1 - avg_chi_sq / 10000)
        
        return results
    
    def histogram_analysis(self, image: np.ndarray) -> Dict:
        """
        Analyze color histogram for anomalies.
        
        LSB modifications create characteristic histogram patterns.
        
        Args:
            image: Image as numpy array
            
        Returns:
            Dictionary with histogram analysis results
        """
        results = {'pair_ratios': [], 'histogram_smoothness': []}
        
        for channel in range(3):
            pixels = image[:, :, channel].flatten()
            hist = np.bincount(pixels, minlength=256)
            
            # Check PoV (Pairs of Values) ratio
            pair_diffs = []
            for i in range(0, 256, 2):
                if hist[i] + hist[i+1] > 0:
                    ratio = abs(hist[i] - hist[i+1]) / (hist[i] + hist[i+1])
                    pair_diffs.append(ratio)
            
            results['pair_ratios'].append(float(np.mean(pair_diffs)) if pair_diffs else 0)
            
            # Check histogram smoothness
            smoothness = np.mean(np.abs(np.diff(hist)))
            results['histogram_smoothness'].append(float(smoothness))
        
        # Low pair ratios indicate steganography
        avg_pair_ratio = np.mean(results['pair_ratios'])
        results['histogram_score'] = max(0, 1 - avg_pair_ratio * 2)
        
        return results
    
    def rs_analysis(self, image: np.ndarray) -> Dict:
        """
        RS (Regular-Singular) steganalysis.
        
        Analyzes groups of pixels for regularity/singularity.
        
        Args:
            image: Image as numpy array
            
        Returns:
            Dictionary with RS analysis results
        """
        results = {'rm_ratio': 0, 'sm_ratio': 0}
        
        # Use grayscale for simplicity
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Group pixels into blocks
        block_size = 4
        h, w = gray.shape
        
        regular_m = 0  # Regular with mask
        singular_m = 0  # Singular with mask
        regular_m_inv = 0  # Regular with inverted mask
        singular_m_inv = 0  # Singular with inverted mask
        
        total_blocks = 0
        
        for y in range(0, h - block_size, block_size):
            for x in range(0, w - block_size, block_size):
                block = gray[y:y+block_size, x:x+block_size].flatten()
                
                # Calculate smoothness function
                f_original = np.sum(np.abs(np.diff(block)))
                
                # Apply mask [1, 0, 1, 0, ...]
                mask = np.array([1, 0] * (len(block) // 2))
                masked = block.copy()
                masked[mask == 1] = block[mask == 1] ^ 1  # Flip LSB
                f_masked = np.sum(np.abs(np.diff(masked)))
                
                # Classify
                if f_masked > f_original:
                    regular_m += 1
                elif f_masked < f_original:
                    singular_m += 1
                
                # Apply inverted mask
                masked_inv = block.copy()
                masked_inv[mask == 0] = block[mask == 0] ^ 1
                f_masked_inv = np.sum(np.abs(np.diff(masked_inv)))
                
                if f_masked_inv > f_original:
                    regular_m_inv += 1
                elif f_masked_inv < f_original:
                    singular_m_inv += 1
                
                total_blocks += 1
        
        if total_blocks > 0:
            results['rm_ratio'] = float(regular_m / total_blocks)
            results['sm_ratio'] = float(singular_m / total_blocks)
            results['rm_inv_ratio'] = float(regular_m_inv / total_blocks)
            results['sm_inv_ratio'] = float(singular_m_inv / total_blocks)
            
            # Calculate RS score
            # Steganography causes R_m ≈ R_{-m} and S_m ≈ S_{-m}
            r_diff = abs(results['rm_ratio'] - results['rm_inv_ratio'])
            s_diff = abs(results['sm_ratio'] - results['sm_inv_ratio'])
            
            # Low differences indicate steganography
            results['rs_score'] = max(0, 1 - (r_diff + s_diff) * 2)
        else:
            results['rs_score'] = 0
        
        return results
    
    def detect(self, image_input: Union[str, bytes, np.ndarray]) -> SteganographyResult:
        """
        Full steganography detection pipeline.
        
        Args:
            image_input: Image to analyze
            
        Returns:
            SteganographyResult with detection results
        """
        image = self.load_image(image_input)
        
        # Run all analyses
        lsb_results = self.analyze_lsb(image)
        chi_results = self.chi_square_analysis(image)
        hist_results = self.histogram_analysis(image)
        rs_results = self.rs_analysis(image)
        
        # Collect scores
        lsb_score = lsb_results.get('lsb_score', 0)
        chi_score = chi_results.get('chi_score', 0)
        hist_score = hist_results.get('histogram_score', 0)
        rs_score = rs_results.get('rs_score', 0)
        
        # Weighted average
        statistical_score = (lsb_score * 0.3 + chi_score * 0.3 + 
                           hist_score * 0.2 + rs_score * 0.2)
        
        # Adjust by sensitivity
        adjusted_threshold = 0.5 * (1 - self.sensitivity * 0.5)
        
        # Determine detection
        hidden_data_detected = statistical_score > adjusted_threshold
        
        # Collect methods that triggered
        detection_methods = []
        if lsb_score > 0.5:
            detection_methods.append("LSB pattern analysis")
        if chi_score > 0.5:
            detection_methods.append("Chi-square analysis")
        if hist_score > 0.5:
            detection_methods.append("Histogram analysis")
        if rs_score > 0.5:
            detection_methods.append("RS steganalysis")
        if lsb_results.get('pattern_detected'):
            detection_methods.append("LSB correlation anomaly")
        
        # Calculate confidence
        if detection_methods:
            confidence = min(1.0, len(detection_methods) / 3)
        else:
            confidence = 1.0 - statistical_score
        
        # Generate explanation
        explanation = self._generate_explanation(
            hidden_data_detected, statistical_score,
            detection_methods, lsb_score, hist_score
        )
        
        return SteganographyResult(
            hidden_data_detected=hidden_data_detected,
            confidence=confidence,
            detection_methods=detection_methods,
            lsb_score=lsb_score,
            histogram_score=hist_score,
            statistical_score=statistical_score,
            explanation=explanation,
        )
    
    def _generate_explanation(
        self,
        detected: bool,
        score: float,
        methods: List[str],
        lsb: float,
        hist: float,
    ) -> str:
        """Generate human-readable explanation."""
        parts = []
        
        if detected:
            parts.append("⚠️ Possible hidden data detected in image!")
        else:
            parts.append("✓ No significant steganographic content detected")
        
        parts.append(f"\nStatistical Score: {score:.2%}")
        parts.append(f"LSB Score: {lsb:.2%}")
        parts.append(f"Histogram Score: {hist:.2%}")
        
        if methods:
            parts.append("\n\nTriggered Detections:")
            for method in methods:
                parts.append(f"  • {method}")
        
        return "\n".join(parts)
