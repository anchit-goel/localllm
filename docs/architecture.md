# System Architecture

## Overview

The Multimodal AI Security System is designed as a modular, extensible framework for detecting various attack vectors targeting AI agents.

## High-Level Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        Input Layer                               │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────────────┐  │
│  │ Audio Input  │  │ Image Input  │  │ Multimodal Input     │  │
│  └──────┬───────┘  └──────┬───────┘  └──────────┬───────────┘  │
└─────────┼─────────────────┼────────────────────┬┼───────────────┘
          │                 │                     │           
          ▼                 ▼                     ▼           
┌─────────────────────────────────────────────────────────────────┐
│                    Processing Layer                              │
│  ┌────────────────────┐  ┌────────────────────┐                │
│  │ Audio Processing   │  │ Visual Processing  │                │
│  │                    │  │                    │                │
│  │ • Feature Extract  │  │ • OCR Detection    │                │
│  │ • Deepfake Detect  │  │ • Injection Anal.  │                │
│  │ • Voice Clone Det. │  │ • Steganography    │                │
│  └─────────┬──────────┘  └─────────┬──────────┘                │
│            │                       │                            │
│            ▼                       ▼                            │
│  ┌────────────────────────────────────────────────────┐        │
│  │           Multimodal Processing                     │        │
│  │           • Consistency Checker                     │        │
│  │           • Cross-Modal Alignment                   │        │
│  └────────────────────┬───────────────────────────────┘        │
└───────────────────────┼─────────────────────────────────────────┘
                        │
                        ▼
┌─────────────────────────────────────────────────────────────────┐
│                    Decision Layer                                │
│  ┌────────────────────┐  ┌────────────────────┐                │
│  │    Risk Engine     │  │    Explainer       │                │
│  │                    │  │                    │                │
│  │ • Score Aggregation│  │ • Report Gen       │                │
│  │ • Threshold Logic  │  │ • Visualization    │                │
│  │ • Action Decision  │  │ • Recommendations  │                │
│  └─────────┬──────────┘  └─────────┬──────────┘                │
└────────────┼───────────────────────┼────────────────────────────┘
             │                       │
             ▼                       ▼
┌─────────────────────────────────────────────────────────────────┐
│                     Output Layer                                 │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────────────┐  │
│  │ Risk Score   │  │ Explanation  │  │ Recommendations      │  │
│  └──────────────┘  └──────────────┘  └──────────────────────┘  │
└─────────────────────────────────────────────────────────────────┘
```

## Module Details

### Audio Processing Module

```
audio/
├── feature_extraction.py    # MFCC, spectral, chroma features
├── deepfake_detector.py     # CNN + LSTM + Ensemble detection
└── voice_clone_detector.py  # Speaker embeddings + replay detection
```

**Feature Extraction Pipeline:**
1. Load audio → Resample to 16kHz
2. Extract MFCC (40 coefficients)
3. Extract spectral features (centroid, rolloff, bandwidth)
4. Generate mel-spectrogram (128 mel bands)
5. Calculate delta/delta-delta for temporal dynamics

**Detection Models:**
- **CNN:** 5 conv blocks processing mel-spectrograms
- **LSTM:** Bidirectional LSTM for temporal patterns
- **Ensemble:** Random Forest + XGBoost + SVM

### Visual Processing Module

```
visual/
├── ocr_detector.py         # Tesseract + EasyOCR text extraction
├── injection_analyzer.py    # Keyword + pattern + anomaly detection
└── steganography_checker.py # LSB + chi-square + RS analysis
```

**OCR Pipeline:**
1. Image preprocessing (contrast enhancement)
2. Text extraction with multiple OCR engines
3. Hidden text detection (low contrast, small fonts)
4. Position anomaly detection

**Injection Detection:**
- Keyword matching (override, ignore, jailbreak, etc.)
- Pattern analysis (role injection, code blocks)
- Visual anomaly detection

### Scoring Module

```
scoring/
├── risk_engine.py   # Multi-factor risk calculation
└── explainer.py     # Report generation
```

**Risk Formula:**
```python
risk = (
    0.4 × audio_deepfake +
    0.3 × visual_injection +
    0.2 × voice_clone +
    0.1 × consistency_mismatch
)
```

## Data Flow

```
1. Input → Validation → Preprocessing
                            ↓
2. Feature Extraction (parallel)
   ├── Audio Features
   └── Visual Features
                            ↓
3. Detection Models
   ├── CNN Prediction
   ├── LSTM Prediction
   ├── ML Ensemble
   ├── OCR Analysis
   └── Steganography
                            ↓
4. Score Aggregation
   ├── Weighted Combination
   └── Threshold Application
                            ↓
5. Decision & Explanation
   ├── Risk Level Classification
   ├── Action Recommendation
   └── Human-Readable Report
```

## Scalability Considerations

### Horizontal Scaling
- Stateless detection modules
- Model inference can be distributed
- API supports batch processing

### Vertical Scaling
- GPU acceleration for CNN/LSTM
- Multi-threading for feature extraction
- Caching for repeated analyses

### Performance Optimization
- Lazy model loading
- Feature extraction parallelization
- Early exit on high-confidence detections
