"""Visual Prompt Injection Analyzer - Part 1: Core functionality."""

import numpy as np
from typing import Dict, Optional, List, Tuple, Union
from dataclasses import dataclass
import logging
import re

try:
    import cv2
except ImportError:
    cv2 = None

from .ocr_detector import OCRDetector, OCRResult

logger = logging.getLogger(__name__)

# Injection keyword categories
INJECTION_KEYWORDS = {
    'override': ['ignore', 'override', 'forget', 'disregard', 'bypass', 'skip'],
    'system_prompt': ['system prompt', 'system message', 'system:', 'base prompt'],
    'jailbreak': ['jailbreak', 'dan', 'developer mode', 'god mode', 'unrestricted'],
    'instruction': ['new instruction', 'new rules', 'instead do', 'real task'],
    'manipulation': ['you are now', 'you must', 'from now on', 'always respond'],
    'context_escape': ['```', '###', '---', 'end of', 'ignore above', 'ignore previous'],
}


@dataclass
class InjectionIndicator:
    """Container for injection indicator."""
    category: str
    matched_text: str
    matched_keyword: str
    confidence: float
    position: Optional[Tuple[int, int, int, int]]
    severity: str
    
    def to_dict(self) -> Dict:
        return {'category': self.category, 'matched_text': self.matched_text,
                'matched_keyword': self.matched_keyword, 'confidence': self.confidence,
                'position': self.position, 'severity': self.severity}


@dataclass
class InjectionResult:
    """Container for injection analysis results."""
    is_malicious: bool
    risk_score: float
    indicators: List[InjectionIndicator]
    extracted_text: str
    hidden_text: str
    visual_anomalies: List[str]
    explanation: str
    confidence: float
    
    def to_dict(self) -> Dict:
        return {'is_malicious': self.is_malicious, 'risk_score': self.risk_score,
                'indicators': [i.to_dict() for i in self.indicators],
                'extracted_text': self.extracted_text, 'hidden_text': self.hidden_text,
                'hidden_text_detected': bool(self.hidden_text),
                'visual_anomalies': self.visual_anomalies, 'explanation': self.explanation,
                'confidence': self.confidence}


class InjectionAnalyzer:
    """Visual prompt injection analyzer."""
    
    def __init__(self, custom_keywords: Optional[Dict] = None):
        self.keywords = INJECTION_KEYWORDS.copy()
        if custom_keywords:
            self.keywords.update(custom_keywords)
        self.severity_weights = {'override': 0.8, 'system_prompt': 0.9, 'jailbreak': 1.0,
                                'instruction': 0.7, 'manipulation': 0.6, 'context_escape': 0.5}
        self.ocr_detector = OCRDetector()
        self._compile_patterns()
        logger.info("InjectionAnalyzer initialized")
    
    def _compile_patterns(self):
        self.patterns = {}
        for category, keywords in self.keywords.items():
            escaped = [re.escape(kw) for kw in keywords]
            pattern = r'\b(' + '|'.join(escaped) + r')\b'
            self.patterns[category] = re.compile(pattern, re.IGNORECASE)
    
    def check_keywords(self, text: str) -> List[InjectionIndicator]:
        indicators = []
        for category, pattern in self.patterns.items():
            for match in pattern.finditer(text):
                weight = self.severity_weights.get(category, 0.5)
                severity = 'critical' if weight >= 0.9 else 'high' if weight >= 0.7 else 'medium'
                start, end = max(0, match.start()-30), min(len(text), match.end()+30)
                indicators.append(InjectionIndicator(
                    category=category, matched_text=text[start:end],
                    matched_keyword=match.group(0), confidence=weight,
                    position=None, severity=severity))
        return indicators
    
    def detect_visual_anomalies(self, image: np.ndarray) -> List[str]:
        anomalies = []
        if cv2 is None:
            return anomalies
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 50, 150)
        h, w = gray.shape
        if np.sum(edges > 0) / (h * w) > 0.15:
            anomalies.append("High edge density - possible hidden patterns")
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        text_regions = sum(1 for c in contours if cv2.boundingRect(c)[2] / (cv2.boundingRect(c)[3]+1) > 5)
        if text_regions > 10:
            anomalies.append(f"Multiple text-like regions detected ({text_regions})")
        return anomalies
    
    def calculate_risk_score(self, indicators: List, hidden: bool, anomalies: List) -> float:
        if not indicators:
            base = 0.0
        else:
            scores = {'critical': 1.0, 'high': 0.7, 'medium': 0.4, 'low': 0.2}
            base = min(1.0, sum(scores.get(i.severity, 0.3) * i.confidence for i in indicators) / 3)
        if hidden:
            base = min(1.0, base + 0.2)
        if anomalies:
            base = min(1.0, base + len(anomalies) * 0.05)
        return base
    
    def analyze(self, image_input: Union[str, bytes, np.ndarray]) -> InjectionResult:
        ocr_result = self.ocr_detector.detect(image_input)
        image = self.ocr_detector.load_image(image_input)
        all_text = ocr_result.full_text + " " + ocr_result.hidden_text
        indicators = self.check_keywords(all_text)
        visual_anomalies = self.detect_visual_anomalies(image)
        visual_anomalies.extend(ocr_result.anomalies)
        risk_score = self.calculate_risk_score(indicators, ocr_result.hidden_text_detected, visual_anomalies)
        is_malicious = risk_score >= 0.5
        confidence = np.mean([i.confidence for i in indicators]) if indicators else 1.0 - risk_score
        
        explanation = "⚠️ PROMPT INJECTION DETECTED" if is_malicious else "✓ No significant threats"
        explanation += f"\nRisk Score: {risk_score:.2%}"
        if indicators:
            explanation += f"\nFound {len(indicators)} suspicious patterns"
        
        return InjectionResult(is_malicious=is_malicious, risk_score=risk_score,
                              indicators=indicators, extracted_text=ocr_result.full_text,
                              hidden_text=ocr_result.hidden_text, visual_anomalies=visual_anomalies,
                              explanation=explanation, confidence=confidence)
