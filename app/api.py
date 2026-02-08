"""
FastAPI REST API for Multimodal Security System.

Provides endpoints for:
- Audio deepfake detection
- Visual injection detection
- Multimodal analysis
- Health checks
"""

import os
import io
import logging
from typing import Optional
from datetime import datetime

from fastapi import FastAPI, File, UploadFile, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel

import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.audio.deepfake_detector import DeepfakeDetector
from src.audio.voice_clone_detector import VoiceCloneDetector
from src.visual.injection_analyzer import InjectionAnalyzer
from src.visual.steganography_checker import SteganographyChecker
from src.multimodal.consistency_checker import ConsistencyChecker
from src.scoring.risk_engine import RiskEngine
from src.scoring.explainer import Explainer

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="Multimodal AI Security API",
    description="Production-ready API for detecting audio deepfakes, voice cloning, and visual prompt injections",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize detectors (lazy loading for performance)
detectors = {}


def get_deepfake_detector():
    if 'deepfake' not in detectors:
        detectors['deepfake'] = DeepfakeDetector()
    return detectors['deepfake']


def get_voice_clone_detector():
    if 'voice_clone' not in detectors:
        detectors['voice_clone'] = VoiceCloneDetector()
    return detectors['voice_clone']


def get_injection_analyzer():
    if 'injection' not in detectors:
        detectors['injection'] = InjectionAnalyzer()
    return detectors['injection']


def get_steganography_checker():
    if 'steganography' not in detectors:
        detectors['steganography'] = SteganographyChecker()
    return detectors['steganography']


def get_consistency_checker():
    if 'consistency' not in detectors:
        detectors['consistency'] = ConsistencyChecker()
    return detectors['consistency']


def get_risk_engine():
    if 'risk' not in detectors:
        detectors['risk'] = RiskEngine()
    return detectors['risk']


def get_explainer():
    if 'explainer' not in detectors:
        detectors['explainer'] = Explainer()
    return detectors['explainer']


# Response models
class HealthResponse(BaseModel):
    status: str
    timestamp: str
    version: str


class AudioDetectionResponse(BaseModel):
    risk_score: float
    classification: str
    confidence: float
    is_fake: bool
    explanation: str
    detected_attacks: list
    model_scores: dict
    processing_time_ms: float


class ImageDetectionResponse(BaseModel):
    injection_detected: bool
    risk_score: float
    extracted_text: str
    hidden_text: str
    suspicious_patterns: list
    visual_anomalies: list
    processing_time_ms: float


class MultimodalResponse(BaseModel):
    overall_risk: float
    risk_level: str
    action: str
    audio_analysis: dict
    visual_analysis: dict
    consistency_score: float
    threats_detected: list
    recommendations: list
    processing_time_ms: float


# Health check endpoint
@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Check API health status."""
    return HealthResponse(
        status="healthy",
        timestamp=datetime.now().isoformat(),
        version="1.0.0"
    )


# Audio detection endpoint
@app.post("/detect/audio", response_model=AudioDetectionResponse)
async def detect_audio(file: UploadFile = File(...)):
    """
    Detect audio deepfakes and voice manipulation.
    
    Accepts WAV, MP3, FLAC audio files.
    Returns risk score, classification, and detailed analysis.
    """
    start_time = datetime.now()
    
    # Validate file type
    allowed_types = ['audio/wav', 'audio/mpeg', 'audio/mp3', 'audio/flac', 'audio/x-wav']
    if file.content_type not in allowed_types:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid file type. Allowed: {allowed_types}"
        )
    
    try:
        # Read file content
        content = await file.read()
        
        # Run detection
        detector = get_deepfake_detector()
        result = detector.detect(content)
        
        # Calculate processing time
        processing_time = (datetime.now() - start_time).total_seconds() * 1000
        
        return AudioDetectionResponse(
            risk_score=result.risk_score,
            classification=detector.get_risk_level(result.risk_score),
            confidence=result.confidence,
            is_fake=result.is_fake,
            explanation=result.explanation,
            detected_attacks=result.detected_artifacts,
            model_scores={
                'cnn': result.cnn_score,
                'ensemble': result.ensemble_score,
                'lstm': result.lstm_score,
            },
            processing_time_ms=processing_time
        )
        
    except Exception as e:
        logger.error(f"Audio detection error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# Image detection endpoint
@app.post("/detect/image", response_model=ImageDetectionResponse)
async def detect_image(file: UploadFile = File(...)):
    """
    Detect visual prompt injections and hidden content.
    
    Accepts PNG, JPG, JPEG image files.
    Returns injection detection results and extracted text.
    """
    start_time = datetime.now()
    
    # Validate file type
    allowed_types = ['image/png', 'image/jpeg', 'image/jpg']
    if file.content_type not in allowed_types:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid file type. Allowed: {allowed_types}"
        )
    
    try:
        # Read file content
        content = await file.read()
        
        # Run injection analysis
        analyzer = get_injection_analyzer()
        result = analyzer.analyze(content)
        
        # Run steganography check
        steg_checker = get_steganography_checker()
        steg_result = steg_checker.detect(content)
        
        # Combine results
        patterns = [ind.to_dict() for ind in result.indicators]
        
        if steg_result.hidden_data_detected:
            patterns.append({
                'category': 'steganography',
                'description': 'Hidden data detected via steganography',
                'confidence': steg_result.confidence,
            })
        
        processing_time = (datetime.now() - start_time).total_seconds() * 1000
        
        return ImageDetectionResponse(
            injection_detected=result.is_malicious,
            risk_score=result.risk_score,
            extracted_text=result.extracted_text[:1000],
            hidden_text=result.hidden_text[:500],
            suspicious_patterns=patterns,
            visual_anomalies=result.visual_anomalies,
            processing_time_ms=processing_time
        )
        
    except Exception as e:
        logger.error(f"Image detection error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# Multimodal detection endpoint
@app.post("/detect/multimodal", response_model=MultimodalResponse)
async def detect_multimodal(
    audio: UploadFile = File(...),
    image: UploadFile = File(...)
):
    """
    Comprehensive multimodal security analysis.
    
    Analyzes both audio and visual content for threats,
    and checks cross-modal consistency.
    """
    start_time = datetime.now()
    
    try:
        # Read files
        audio_content = await audio.read()
        image_content = await image.read()
        
        # Run audio analysis
        deepfake_detector = get_deepfake_detector()
        audio_result = deepfake_detector.detect(audio_content)
        
        # Run voice clone detection
        voice_detector = get_voice_clone_detector()
        voice_result = voice_detector.detect(audio_content)
        
        # Run image analysis
        injection_analyzer = get_injection_analyzer()
        visual_result = injection_analyzer.analyze(image_content)
        
        # Check consistency
        consistency_checker = get_consistency_checker()
        consistency_result = consistency_checker.check(
            audio_content=audio_result.explanation,
            visual_content=visual_result.extracted_text,
        )
        
        # Calculate overall risk
        risk_engine = get_risk_engine()
        assessment = risk_engine.assess(
            deepfake_result=audio_result.to_dict(),
            voice_clone_result=voice_result.to_dict(),
            injection_result=visual_result.to_dict(),
            consistency_result=consistency_result.to_dict(),
        )
        
        processing_time = (datetime.now() - start_time).total_seconds() * 1000
        
        return MultimodalResponse(
            overall_risk=assessment.overall_risk,
            risk_level=assessment.risk_level.value,
            action=assessment.action,
            audio_analysis={
                'deepfake_risk': audio_result.risk_score,
                'voice_clone_risk': voice_result.similarity_score,
                'is_fake': audio_result.is_fake,
            },
            visual_analysis={
                'injection_risk': visual_result.risk_score,
                'is_malicious': visual_result.is_malicious,
                'hidden_content': visual_result.hidden_text_detected,
            },
            consistency_score=consistency_result.consistency_score,
            threats_detected=assessment.threats_detected,
            recommendations=assessment.recommendations,
            processing_time_ms=processing_time
        )
        
    except Exception as e:
        logger.error(f"Multimodal detection error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# Batch processing endpoint
@app.post("/detect/batch")
async def detect_batch(files: list[UploadFile] = File(...)):
    """
    Process multiple files in batch.
    
    Returns results for each file.
    """
    results = []
    
    for file in files:
        try:
            content = await file.read()
            
            # Determine file type and process accordingly
            if file.content_type and file.content_type.startswith('audio'):
                detector = get_deepfake_detector()
                result = detector.detect(content)
                results.append({
                    'filename': file.filename,
                    'type': 'audio',
                    'risk_score': result.risk_score,
                    'is_fake': result.is_fake,
                })
            elif file.content_type and file.content_type.startswith('image'):
                analyzer = get_injection_analyzer()
                result = analyzer.analyze(content)
                results.append({
                    'filename': file.filename,
                    'type': 'image',
                    'risk_score': result.risk_score,
                    'is_malicious': result.is_malicious,
                })
            else:
                results.append({
                    'filename': file.filename,
                    'error': 'Unsupported file type',
                })
                
        except Exception as e:
            results.append({
                'filename': file.filename,
                'error': str(e),
            })
    
    return {'results': results}


# Report generation endpoint
@app.post("/report")
async def generate_report(
    audio: Optional[UploadFile] = File(None),
    image: Optional[UploadFile] = File(None),
):
    """
    Generate detailed security report.
    
    Returns comprehensive HTML report.
    """
    try:
        results = {}
        
        if audio:
            audio_content = await audio.read()
            detector = get_deepfake_detector()
            results['audio'] = detector.detect(audio_content).to_dict()
        
        if image:
            image_content = await image.read()
            analyzer = get_injection_analyzer()
            results['visual'] = analyzer.analyze(image_content).to_dict()
        
        # Generate risk assessment
        risk_engine = get_risk_engine()
        assessment = risk_engine.assess(
            deepfake_result=results.get('audio'),
            injection_result=results.get('visual'),
        )
        
        # Generate report
        explainer = get_explainer()
        report = explainer.generate_report(
            risk_assessment=assessment.to_dict(),
            audio_result=results.get('audio'),
            visual_result=results.get('visual'),
        )
        
        return {
            'report': report.to_dict(),
            'html': report.to_html(),
        }
        
    except Exception as e:
        logger.error(f"Report generation error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


def main():
    """Run the API server."""
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)


if __name__ == "__main__":
    main()
