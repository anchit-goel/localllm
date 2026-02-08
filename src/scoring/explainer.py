"""
Explainability Engine.

Generates human-readable reports and visualizations:
- Detection reports
- Feature importance
- Anomaly highlighting
- Confidence breakdowns
"""

import numpy as np
from typing import Dict, Optional, List, Any
from dataclasses import dataclass
import logging
from datetime import datetime

logger = logging.getLogger(__name__)


@dataclass
class ExplanationReport:
    """Container for explanation report."""
    summary: str
    risk_level: str
    confidence_breakdown: Dict[str, float]
    detected_threats: List[Dict]
    feature_importance: Dict[str, float]
    recommendations: List[str]
    timestamp: str
    metadata: Dict[str, Any]
    
    def to_dict(self) -> Dict:
        return {
            'summary': self.summary,
            'risk_level': self.risk_level,
            'confidence_breakdown': self.confidence_breakdown,
            'detected_threats': self.detected_threats,
            'feature_importance': self.feature_importance,
            'recommendations': self.recommendations,
            'timestamp': self.timestamp,
            'metadata': self.metadata,
        }
    
    def to_html(self) -> str:
        """Generate HTML report."""
        html = f"""
        <html>
        <head>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 20px; }}
                .header {{ background: #2c3e50; color: white; padding: 20px; }}
                .section {{ margin: 20px 0; padding: 15px; border: 1px solid #ddd; }}
                .risk-high {{ color: #e74c3c; }}
                .risk-medium {{ color: #f39c12; }}
                .risk-low {{ color: #27ae60; }}
                .threat {{ background: #ffe6e6; padding: 10px; margin: 5px 0; }}
                table {{ width: 100%; border-collapse: collapse; }}
                th, td {{ padding: 10px; border: 1px solid #ddd; text-align: left; }}
            </style>
        </head>
        <body>
            <div class="header">
                <h1>Security Analysis Report</h1>
                <p>Generated: {self.timestamp}</p>
            </div>
            
            <div class="section">
                <h2>Summary</h2>
                <p>{self.summary}</p>
                <p><strong>Risk Level:</strong> 
                   <span class="risk-{self.risk_level.lower()}">{self.risk_level}</span>
                </p>
            </div>
            
            <div class="section">
                <h2>Confidence Breakdown</h2>
                <table>
                    <tr><th>Component</th><th>Score</th></tr>
                    {''.join(f'<tr><td>{k}</td><td>{v:.1%}</td></tr>' 
                             for k, v in self.confidence_breakdown.items())}
                </table>
            </div>
            
            <div class="section">
                <h2>Detected Threats</h2>
                {''.join(f'<div class="threat"><strong>{t.get("type", "Unknown")}</strong>: {t.get("description", "")}</div>' 
                         for t in self.detected_threats) or '<p>No threats detected</p>'}
            </div>
            
            <div class="section">
                <h2>Recommendations</h2>
                <ul>
                    {''.join(f'<li>{r}</li>' for r in self.recommendations)}
                </ul>
            </div>
        </body>
        </html>
        """
        return html


class Explainer:
    """
    Explainability engine for security analysis.
    
    Provides:
    - Human-readable reports
    - Visualization generation
    - Confidence explanations
    - Feature attribution
    """
    
    def __init__(self, detail_level: str = 'medium'):
        """
        Initialize explainer.
        
        Args:
            detail_level: 'brief', 'medium', or 'detailed'
        """
        self.detail_level = detail_level
        logger.info(f"Explainer initialized with {detail_level} detail level")
    
    def generate_report(
        self,
        risk_assessment: Dict,
        audio_result: Optional[Dict] = None,
        visual_result: Optional[Dict] = None,
        consistency_result: Optional[Dict] = None,
        metadata: Optional[Dict] = None,
    ) -> ExplanationReport:
        """
        Generate comprehensive explanation report.
        
        Args:
            risk_assessment: Risk engine assessment result
            audio_result: Audio detection results
            visual_result: Visual detection results
            consistency_result: Consistency check results
            metadata: Additional metadata
            
        Returns:
            ExplanationReport with full analysis
        """
        # Generate summary
        summary = self._generate_summary(risk_assessment)
        
        # Extract risk level
        risk_level = risk_assessment.get('risk_level', 'UNKNOWN')
        
        # Build confidence breakdown
        confidence = {
            'Overall': risk_assessment.get('overall_risk', 0),
            'Audio Analysis': risk_assessment.get('audio_risk', 0),
            'Visual Analysis': risk_assessment.get('visual_risk', 0),
            'Consistency': 1 - risk_assessment.get('consistency_risk', 0),
        }
        
        # Collect threats with details
        threats = []
        for threat in risk_assessment.get('threats_detected', []):
            threats.append({
                'type': self._categorize_threat(threat),
                'description': threat,
                'severity': self._get_threat_severity(threat),
            })
        
        # Add audio-specific threats
        if audio_result:
            if audio_result.get('detected_artifacts'):
                for artifact in audio_result['detected_artifacts']:
                    threats.append({
                        'type': 'Audio Artifact',
                        'description': artifact,
                        'severity': 'medium',
                    })
        
        # Add visual-specific threats
        if visual_result:
            if visual_result.get('indicators'):
                for ind in visual_result.get('indicators', [])[:5]:
                    threats.append({
                        'type': 'Visual Injection',
                        'description': f"{ind.get('category', 'Unknown')}: {ind.get('matched_keyword', '')}",
                        'severity': ind.get('severity', 'medium'),
                    })
        
        # Get feature importance
        feature_importance = {}
        if audio_result and audio_result.get('feature_importance'):
            feature_importance.update(audio_result['feature_importance'])
        
        # Get recommendations
        recommendations = risk_assessment.get('recommendations', [])
        
        return ExplanationReport(
            summary=summary,
            risk_level=risk_level,
            confidence_breakdown=confidence,
            detected_threats=threats,
            feature_importance=feature_importance,
            recommendations=recommendations,
            timestamp=datetime.now().isoformat(),
            metadata=metadata or {},
        )
    
    def _generate_summary(self, assessment: Dict) -> str:
        """Generate summary text."""
        risk = assessment.get('overall_risk', 0)
        level = assessment.get('risk_level', 'UNKNOWN')
        threats = assessment.get('threats_detected', [])
        
        if self.detail_level == 'brief':
            if risk < 0.3:
                return f"Content appears safe. Risk: {risk:.1%}"
            elif risk < 0.6:
                return f"Potential threats detected. Risk: {risk:.1%}"
            else:
                return f"HIGH RISK content detected. Risk: {risk:.1%}"
        
        # Medium/detailed summary
        parts = []
        
        if risk < 0.3:
            parts.append("The analyzed content appears to be genuine and safe.")
            parts.append(f"Overall risk assessment: {risk:.1%} ({level})")
        elif risk < 0.6:
            parts.append("The content shows some suspicious characteristics that warrant caution.")
            parts.append(f"Overall risk assessment: {risk:.1%} ({level})")
            if threats:
                parts.append(f"Detected {len(threats)} potential threat(s).")
        else:
            parts.append("⚠️ HIGH RISK: The content shows strong indicators of manipulation or attack.")
            parts.append(f"Overall risk assessment: {risk:.1%} ({level})")
            parts.append(f"Detected {len(threats)} significant threat(s).")
        
        if self.detail_level == 'detailed' and threats:
            parts.append("\nPrimary concerns:")
            for threat in threats[:3]:
                parts.append(f"  • {threat}")
        
        return '\n'.join(parts)
    
    def _categorize_threat(self, threat_str: str) -> str:
        """Categorize threat by type."""
        threat_lower = threat_str.lower()
        if 'deepfake' in threat_lower:
            return 'Deepfake Detection'
        elif 'voice' in threat_lower or 'clone' in threat_lower:
            return 'Voice Cloning'
        elif 'injection' in threat_lower:
            return 'Prompt Injection'
        elif 'consistency' in threat_lower:
            return 'Cross-Modal Mismatch'
        elif 'steganograph' in threat_lower:
            return 'Hidden Data'
        else:
            return 'General Threat'
    
    def _get_threat_severity(self, threat_str: str) -> str:
        """Determine threat severity."""
        if any(word in threat_str.lower() for word in ['high', 'critical', 'severe']):
            return 'high'
        elif any(word in threat_str.lower() for word in ['medium', 'moderate']):
            return 'medium'
        else:
            return 'low'
    
    def explain_audio_detection(self, audio_result: Dict) -> str:
        """Generate explanation for audio detection results."""
        parts = ["Audio Analysis Explanation", "=" * 30]
        
        risk = audio_result.get('risk_score', 0)
        parts.append(f"\nDeepfake Risk Score: {risk:.1%}")
        
        if risk < 0.3:
            parts.append("The audio appears to be genuine with high confidence.")
        elif risk < 0.6:
            parts.append("The audio shows some suspicious characteristics.")
        else:
            parts.append("The audio is likely synthetic or manipulated.")
        
        # Model scores
        parts.append(f"\nModel Scores:")
        parts.append(f"  CNN Spectrogram: {audio_result.get('cnn_score', 0):.1%}")
        parts.append(f"  ML Ensemble: {audio_result.get('ensemble_score', 0):.1%}")
        parts.append(f"  LSTM Temporal: {audio_result.get('lstm_score', 0):.1%}")
        
        # Artifacts
        artifacts = audio_result.get('detected_artifacts', [])
        if artifacts:
            parts.append(f"\nDetected Artifacts:")
            for a in artifacts:
                parts.append(f"  • {a}")
        
        return '\n'.join(parts)
    
    def explain_visual_detection(self, visual_result: Dict) -> str:
        """Generate explanation for visual detection results."""
        parts = ["Visual Analysis Explanation", "=" * 30]
        
        risk = visual_result.get('risk_score', 0)
        parts.append(f"\nInjection Risk Score: {risk:.1%}")
        
        is_malicious = visual_result.get('is_malicious', False)
        if is_malicious:
            parts.append("⚠️ Prompt injection attempt detected!")
        else:
            parts.append("No significant injection threats found.")
        
        # Extracted text
        text = visual_result.get('extracted_text', '')
        if text:
            parts.append(f"\nExtracted Text Preview:")
            parts.append(f"  '{text[:200]}...'")
        
        # Hidden text
        hidden = visual_result.get('hidden_text', '')
        if hidden:
            parts.append(f"\n🔴 Hidden Text Found:")
            parts.append(f"  '{hidden[:100]}'")
        
        # Indicators
        indicators = visual_result.get('indicators', [])
        if indicators:
            parts.append(f"\nDetection Indicators:")
            for ind in indicators[:5]:
                if isinstance(ind, dict):
                    parts.append(f"  [{ind.get('severity', '?').upper()}] {ind.get('category', 'Unknown')}")
        
        return '\n'.join(parts)
    
    def generate_visualization_data(
        self,
        risk_assessment: Dict,
    ) -> Dict:
        """
        Generate data for visualizations.
        
        Returns data suitable for plotting with matplotlib/plotly.
        
        Args:
            risk_assessment: Risk engine assessment
            
        Returns:
            Dictionary with visualization data
        """
        components = risk_assessment.get('component_scores', {})
        
        return {
            'risk_gauge': {
                'value': risk_assessment.get('overall_risk', 0),
                'max': 1.0,
                'thresholds': [0.3, 0.6, 0.8],
                'colors': ['green', 'yellow', 'orange', 'red'],
            },
            'component_bar': {
                'labels': list(components.keys()),
                'values': list(components.values()),
                'colors': ['red' if v > 0.5 else 'green' for v in components.values()],
            },
            'threat_count': {
                'detected': len(risk_assessment.get('threats_detected', [])),
                'categories': self._count_threat_categories(
                    risk_assessment.get('threats_detected', [])
                ),
            },
        }
    
    def _count_threat_categories(self, threats: List[str]) -> Dict[str, int]:
        """Count threats by category."""
        categories = {}
        for threat in threats:
            cat = self._categorize_threat(threat)
            categories[cat] = categories.get(cat, 0) + 1
        return categories
