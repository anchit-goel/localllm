"""
Risk Scoring Engine.

Combines detection results from all modules into unified risk assessment:
- Multi-factor risk scoring
- Decision logic (PASS/FLAG/BLOCK/ALERT)
- Threshold-based actions
"""

import numpy as np
from typing import Dict, Optional, List, Union
from dataclasses import dataclass
from enum import Enum
import logging

logger = logging.getLogger(__name__)


class RiskLevel(Enum):
    """Risk level classifications."""
    PASS = "PASS"      # 0.0-0.3: Green - allow
    FLAG = "FLAG"      # 0.3-0.6: Yellow - warn but allow
    BLOCK = "BLOCK"    # 0.6-0.8: Orange - reject with explanation
    ALERT = "ALERT"    # 0.8-1.0: Red - block + log for security review


@dataclass
class RiskAssessment:
    """Container for risk assessment results."""
    overall_risk: float
    risk_level: RiskLevel
    audio_risk: float
    visual_risk: float
    consistency_risk: float
    component_scores: Dict[str, float]
    threats_detected: List[str]
    action: str
    explanation: str
    recommendations: List[str]
    
    def to_dict(self) -> Dict:
        return {
            'overall_risk': self.overall_risk,
            'risk_level': self.risk_level.value,
            'audio_risk': self.audio_risk,
            'visual_risk': self.visual_risk,
            'consistency_risk': self.consistency_risk,
            'component_scores': self.component_scores,
            'threats_detected': self.threats_detected,
            'action': self.action,
            'explanation': self.explanation,
            'recommendations': self.recommendations,
        }


class RiskEngine:
    """
    Multi-factor risk scoring and decision engine.
    
    Aggregates results from:
    - Audio deepfake detection
    - Voice cloning detection
    - Visual injection detection
    - Cross-modal consistency
    """
    
    def __init__(
        self,
        audio_weight: float = 0.4,
        visual_weight: float = 0.3,
        voice_clone_weight: float = 0.2,
        consistency_weight: float = 0.1,
        thresholds: Optional[Dict[str, float]] = None,
    ):
        """
        Initialize risk engine.
        
        Args:
            audio_weight: Weight for audio deepfake score
            visual_weight: Weight for visual injection score
            voice_clone_weight: Weight for voice cloning score
            consistency_weight: Weight for cross-modal mismatch
            thresholds: Custom risk level thresholds
        """
        self.weights = {
            'audio_deepfake': audio_weight,
            'visual_injection': visual_weight,
            'voice_clone': voice_clone_weight,
            'consistency': consistency_weight,
        }
        
        # Normalize weights
        total = sum(self.weights.values())
        self.weights = {k: v/total for k, v in self.weights.items()}
        
        # Risk thresholds
        self.thresholds = thresholds or {
            'pass_max': 0.3,
            'flag_max': 0.6,
            'block_max': 0.8,
        }
        
        logger.info(f"RiskEngine initialized with weights: {self.weights}")
    
    def calculate_risk(
        self,
        audio_deepfake_score: float = 0.0,
        voice_clone_score: float = 0.0,
        visual_injection_score: float = 0.0,
        consistency_mismatch: float = 0.0,
        additional_factors: Optional[Dict[str, float]] = None,
    ) -> float:
        """
        Calculate overall risk score.
        
        Formula:
        risk = w1*audio + w2*visual + w3*voice + w4*consistency + adjustments
        
        Args:
            audio_deepfake_score: Deepfake detection confidence (0-1)
            voice_clone_score: Voice cloning similarity (0-1)
            visual_injection_score: Injection detection score (0-1)
            consistency_mismatch: Cross-modal mismatch (0-1)
            additional_factors: Extra scoring factors
            
        Returns:
            Overall risk score (0-1)
        """
        # Base weighted score
        risk_score = (
            self.weights['audio_deepfake'] * audio_deepfake_score +
            self.weights['voice_clone'] * voice_clone_score +
            self.weights['visual_injection'] * visual_injection_score +
            self.weights['consistency'] * (1 - consistency_mismatch)
        )
        
        # Apply additional factors
        if additional_factors:
            for factor, score in additional_factors.items():
                risk_score = max(risk_score, score * 0.5)
        
        # Cap at 1.0
        return min(1.0, max(0.0, risk_score))
    
    def get_risk_level(self, risk_score: float) -> RiskLevel:
        """
        Get risk level classification from score.
        
        Args:
            risk_score: Overall risk score (0-1)
            
        Returns:
            RiskLevel enum value
        """
        if risk_score <= self.thresholds['pass_max']:
            return RiskLevel.PASS
        elif risk_score <= self.thresholds['flag_max']:
            return RiskLevel.FLAG
        elif risk_score <= self.thresholds['block_max']:
            return RiskLevel.BLOCK
        else:
            return RiskLevel.ALERT
    
    def get_action(self, risk_level: RiskLevel) -> str:
        """
        Get recommended action for risk level.
        
        Args:
            risk_level: Risk level classification
            
        Returns:
            Action string
        """
        actions = {
            RiskLevel.PASS: "ALLOW - Content appears safe",
            RiskLevel.FLAG: "WARN - Allow with caution flag",
            RiskLevel.BLOCK: "REJECT - Block with explanation",
            RiskLevel.ALERT: "BLOCK + ALERT - Security review required",
        }
        return actions.get(risk_level, "UNKNOWN")
    
    def get_recommendations(
        self,
        risk_level: RiskLevel,
        threats: List[str],
    ) -> List[str]:
        """
        Generate recommendations based on risk assessment.
        
        Args:
            risk_level: Current risk level
            threats: List of detected threats
            
        Returns:
            List of recommendations
        """
        recommendations = []
        
        if risk_level == RiskLevel.PASS:
            recommendations.append("No immediate action required")
            recommendations.append("Continue monitoring for pattern changes")
        
        elif risk_level == RiskLevel.FLAG:
            recommendations.append("Review flagged content before processing")
            recommendations.append("Consider additional verification for sensitive operations")
            if any('deepfake' in t.lower() for t in threats):
                recommendations.append("Request alternative voice verification")
        
        elif risk_level == RiskLevel.BLOCK:
            recommendations.append("Do not process this content")
            recommendations.append("Request authentic content from verified source")
            if any('injection' in t.lower() for t in threats):
                recommendations.append("Report potential injection attempt")
        
        elif risk_level == RiskLevel.ALERT:
            recommendations.append("Immediately escalate to security team")
            recommendations.append("Log all details for forensic analysis")
            recommendations.append("Block source from further submissions")
            recommendations.append("Review related submissions for patterns")
        
        return recommendations
    
    def assess(
        self,
        deepfake_result: Optional[Dict] = None,
        voice_clone_result: Optional[Dict] = None,
        injection_result: Optional[Dict] = None,
        consistency_result: Optional[Dict] = None,
        steganography_result: Optional[Dict] = None,
    ) -> RiskAssessment:
        """
        Perform full risk assessment from detection results.
        
        Args:
            deepfake_result: Deepfake detection result dict
            voice_clone_result: Voice clone detection result dict
            injection_result: Injection detection result dict
            consistency_result: Consistency check result dict
            steganography_result: Steganography detection result dict
            
        Returns:
            RiskAssessment with full analysis
        """
        # Extract scores with defaults
        audio_deepfake = (deepfake_result or {}).get('risk_score', 0.0)
        voice_clone = (voice_clone_result or {}).get('similarity_score', 0.0)
        if voice_clone > 0.8:  # High similarity could indicate cloning
            voice_clone = 1 - voice_clone  # Invert for risk
        visual_injection = (injection_result or {}).get('risk_score', 0.0)
        consistency = (consistency_result or {}).get('consistency_score', 1.0)
        
        # Additional factors
        additional = {}
        if steganography_result:
            if steganography_result.get('hidden_data_detected'):
                additional['steganography'] = steganography_result.get('confidence', 0.5)
        
        # Calculate overall risk
        overall_risk = self.calculate_risk(
            audio_deepfake_score=audio_deepfake,
            voice_clone_score=voice_clone,
            visual_injection_score=visual_injection,
            consistency_mismatch=1 - consistency,
            additional_factors=additional,
        )
        
        # Get risk level and action
        risk_level = self.get_risk_level(overall_risk)
        action = self.get_action(risk_level)
        
        # Collect threats
        threats = []
        if audio_deepfake > 0.5:
            threats.append(f"Audio deepfake detected (confidence: {audio_deepfake:.1%})")
        if voice_clone > 0.5:
            threats.append(f"Potential voice cloning (score: {voice_clone:.1%})")
        if visual_injection > 0.5:
            threats.append(f"Visual prompt injection (risk: {visual_injection:.1%})")
        if consistency < 0.5:
            threats.append(f"Cross-modal inconsistency (score: {consistency:.1%})")
        if steganography_result and steganography_result.get('hidden_data_detected'):
            threats.append("Steganographic content detected")
        
        # Get recommendations
        recommendations = self.get_recommendations(risk_level, threats)
        
        # Component scores
        component_scores = {
            'audio_deepfake': audio_deepfake,
            'voice_clone': voice_clone,
            'visual_injection': visual_injection,
            'consistency': consistency,
        }
        if additional:
            component_scores.update(additional)
        
        # Generate explanation
        explanation = self._generate_explanation(
            overall_risk, risk_level, threats, component_scores
        )
        
        return RiskAssessment(
            overall_risk=overall_risk,
            risk_level=risk_level,
            audio_risk=max(audio_deepfake, voice_clone),
            visual_risk=visual_injection,
            consistency_risk=1 - consistency,
            component_scores=component_scores,
            threats_detected=threats,
            action=action,
            explanation=explanation,
            recommendations=recommendations,
        )
    
    def _generate_explanation(
        self,
        overall: float,
        level: RiskLevel,
        threats: List[str],
        components: Dict[str, float],
    ) -> str:
        """Generate human-readable explanation."""
        # Risk level colors
        level_indicators = {
            RiskLevel.PASS: "🟢 PASS",
            RiskLevel.FLAG: "🟡 FLAG",
            RiskLevel.BLOCK: "🟠 BLOCK",
            RiskLevel.ALERT: "🔴 ALERT",
        }
        
        parts = [
            f"Risk Assessment: {level_indicators.get(level, level.value)}",
            f"Overall Risk Score: {overall:.1%}",
            "",
            "Component Scores:",
        ]
        
        for component, score in components.items():
            indicator = "⚠️" if score > 0.5 else "✓"
            parts.append(f"  {indicator} {component}: {score:.1%}")
        
        if threats:
            parts.extend(["", "Detected Threats:"])
            for threat in threats:
                parts.append(f"  • {threat}")
        
        return "\n".join(parts)
