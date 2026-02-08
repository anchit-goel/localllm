"""Multimodal AI Security System - Protecting AI agents from deepfakes and prompt injections."""

__version__ = "1.0.0"
__author__ = "Multimodal Security Team"

from .audio.feature_extraction import AudioFeatureExtractor
from .audio.deepfake_detector import DeepfakeDetector
from .audio.voice_clone_detector import VoiceCloneDetector
from .visual.ocr_detector import OCRDetector
from .visual.injection_analyzer import InjectionAnalyzer
from .visual.steganography_checker import SteganographyChecker
from .multimodal.consistency_checker import ConsistencyChecker
from .scoring.risk_engine import RiskEngine
from .scoring.explainer import Explainer

__all__ = [
    "AudioFeatureExtractor",
    "DeepfakeDetector",
    "VoiceCloneDetector",
    "OCRDetector",
    "InjectionAnalyzer",
    "SteganographyChecker",
    "ConsistencyChecker",
    "RiskEngine",
    "Explainer",
]
