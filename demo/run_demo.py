#!/usr/bin/env python
"""
Demo script for Multimodal AI Security System.

Demonstrates the capabilities of the detection system with sample data.
"""

import os
import sys
import numpy as np
import tempfile

# Add project root to path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

from src.audio.deepfake_detector import DeepfakeDetector
from src.audio.voice_clone_detector import VoiceCloneDetector
from src.visual.injection_analyzer import InjectionAnalyzer
from src.visual.steganography_checker import SteganographyChecker
from src.scoring.risk_engine import RiskEngine
from src.scoring.explainer import Explainer


def generate_synthetic_audio(duration: float = 3.0, sample_rate: int = 16000) -> np.ndarray:
    """Generate synthetic audio for demo."""
    t = np.linspace(0, duration, int(sample_rate * duration), endpoint=False)
    
    # Create speech-like audio with varying frequency
    freq = 150 + 50 * np.sin(2 * np.pi * 0.5 * t)
    waveform = 0.5 * np.sin(2 * np.pi * freq * t)
    
    # Add harmonics
    for harmonic in [2, 3, 4]:
        waveform += (0.3 / harmonic) * np.sin(2 * np.pi * harmonic * freq * t)
    
    # Add noise
    waveform += 0.02 * np.random.randn(len(waveform))
    
    return waveform.astype(np.float32)


def generate_sample_image(width: int = 640, height: int = 480) -> np.ndarray:
    """Generate sample image for demo."""
    try:
        import cv2
        
        # Create gradient background
        image = np.zeros((height, width, 3), dtype=np.uint8)
        for i in range(height):
            color = int(200 * (i / height))
            image[i, :] = [color, color // 2, 255 - color]
        
        # Add some shapes
        cv2.circle(image, (width // 2, height // 2), 100, (255, 255, 255), -1)
        cv2.rectangle(image, (50, 50), (150, 150), (0, 255, 0), 2)
        
        return image
    except ImportError:
        return np.random.randint(100, 200, (height, width, 3), dtype=np.uint8)


def demo_audio_detection():
    """Demo audio deepfake detection."""
    print("\n" + "=" * 60)
    print("🎵 AUDIO DEEPFAKE DETECTION DEMO")
    print("=" * 60)
    
    # Initialize detector
    print("\n[1] Initializing audio deepfake detector...")
    detector = DeepfakeDetector()
    
    # Generate sample audio
    print("[2] Generating sample audio (3 seconds)...")
    waveform = generate_synthetic_audio()
    
    # Run detection
    print("[3] Running detection pipeline...")
    result = detector.detect(waveform, sample_rate=16000)
    
    # Display results
    print(f"\n📊 RESULTS:")
    print(f"   Risk Score: {result.risk_score:.1%}")
    print(f"   Classification: {detector.get_risk_level(result.risk_score)}")
    print(f"   Is Fake: {'Yes' if result.is_fake else 'No'}")
    print(f"   Confidence: {result.confidence:.1%}")
    
    print(f"\n📈 Model Scores:")
    print(f"   CNN Spectrogram: {result.cnn_score:.1%}")
    print(f"   ML Ensemble: {result.ensemble_score:.1%}")
    print(f"   LSTM Temporal: {result.lstm_score:.1%}")
    
    if result.detected_artifacts:
        print(f"\n⚠️ Detected Artifacts:")
        for artifact in result.detected_artifacts:
            print(f"   • {artifact}")
    
    print(f"\n💬 Explanation:")
    print(f"   {result.explanation[:200]}...")


def demo_visual_detection():
    """Demo visual injection detection."""
    print("\n" + "=" * 60)
    print("🖼️ VISUAL INJECTION DETECTION DEMO")
    print("=" * 60)
    
    # Initialize analyzer
    print("\n[1] Initializing injection analyzer...")
    analyzer = InjectionAnalyzer()
    
    # Generate sample image
    print("[2] Generating sample image...")
    image = generate_sample_image()
    
    # Run detection
    print("[3] Running injection detection...")
    result = analyzer.analyze(image)
    
    # Display results
    print(f"\n📊 RESULTS:")
    print(f"   Risk Score: {result.risk_score:.1%}")
    print(f"   Is Malicious: {'Yes' if result.is_malicious else 'No'}")
    print(f"   Indicators Found: {len(result.indicators)}")
    print(f"   Hidden Text Detected: {'Yes' if result.hidden_text_detected else 'No'}")
    
    if result.extracted_text:
        print(f"\n📝 Extracted Text:")
        print(f"   {result.extracted_text[:100] if result.extracted_text else 'None'}...")
    
    print(f"\n💬 Explanation:")
    print(f"   {result.explanation}")


def demo_steganography_detection():
    """Demo steganography detection."""
    print("\n" + "=" * 60)
    print("🔐 STEGANOGRAPHY DETECTION DEMO")
    print("=" * 60)
    
    # Initialize checker
    print("\n[1] Initializing steganography checker...")
    checker = SteganographyChecker()
    
    # Generate sample image
    print("[2] Generating sample image...")
    image = generate_sample_image()
    
    # Run detection
    print("[3] Running steganography analysis...")
    result = checker.detect(image)
    
    # Display results
    print(f"\n📊 RESULTS:")
    print(f"   Hidden Data Detected: {'Yes' if result.hidden_data_detected else 'No'}")
    print(f"   Confidence: {result.confidence:.1%}")
    print(f"   LSB Score: {result.lsb_score:.1%}")
    print(f"   Histogram Score: {result.histogram_score:.1%}")
    print(f"   Statistical Score: {result.statistical_score:.1%}")
    
    if result.detection_methods:
        print(f"\n🔍 Detection Methods Triggered:")
        for method in result.detection_methods:
            print(f"   • {method}")


def demo_risk_scoring():
    """Demo risk scoring engine."""
    print("\n" + "=" * 60)
    print("📊 RISK SCORING ENGINE DEMO")
    print("=" * 60)
    
    # Initialize components
    print("\n[1] Initializing risk engine...")
    engine = RiskEngine()
    explainer = Explainer()
    
    # Create mock results
    print("[2] Creating sample detection results...")
    mock_audio = {'risk_score': 0.65, 'is_fake': True}
    mock_visual = {'risk_score': 0.40, 'is_malicious': False}
    mock_consistency = {'consistency_score': 0.75}
    
    # Run risk assessment
    print("[3] Running risk assessment...")
    assessment = engine.assess(
        deepfake_result=mock_audio,
        injection_result=mock_visual,
        consistency_result=mock_consistency,
    )
    
    # Display results
    print(f"\n📊 RISK ASSESSMENT:")
    print(f"   Overall Risk: {assessment.overall_risk:.1%}")
    print(f"   Risk Level: {assessment.risk_level.value}")
    print(f"   Audio Risk: {assessment.audio_risk:.1%}")
    print(f"   Visual Risk: {assessment.visual_risk:.1%}")
    print(f"   Consistency Risk: {assessment.consistency_risk:.1%}")
    
    print(f"\n🎯 Action: {assessment.action}")
    
    if assessment.threats_detected:
        print(f"\n⚠️ Threats Detected:")
        for threat in assessment.threats_detected:
            print(f"   • {threat}")
    
    print(f"\n💡 Recommendations:")
    for rec in assessment.recommendations:
        print(f"   • {rec}")
    
    # Generate report
    print("\n[4] Generating explanation report...")
    report = explainer.generate_report(assessment.to_dict())
    print(f"\n📄 Report Summary:")
    print(f"   {report.summary}")


def main():
    """Run all demos."""
    print("\n" + "=" * 60)
    print("🛡️ MULTIMODAL AI SECURITY SYSTEM - DEMO")
    print("=" * 60)
    print("\nThis demo showcases the detection capabilities of the system.")
    print("Note: Using synthetic data for demonstration purposes.")
    
    # Run demos
    demo_audio_detection()
    demo_visual_detection()
    demo_steganography_detection()
    demo_risk_scoring()
    
    print("\n" + "=" * 60)
    print("✅ DEMO COMPLETE")
    print("=" * 60)
    print("\nTo run the full system:")
    print("  • API:       uvicorn app.api:app --reload")
    print("  • Web UI:    streamlit run app/streamlit_app.py")
    print("  • Tests:     pytest tests/ -v")
    print()


if __name__ == "__main__":
    main()
