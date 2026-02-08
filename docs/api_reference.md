# API Reference

## Base URL

```
http://localhost:8000
```

## Authentication

Currently, no authentication is required. For production, implement JWT or API key authentication.

---

## Endpoints

### Health Check

Check API status.

```http
GET /health
```

**Response:**
```json
{
  "status": "healthy",
  "timestamp": "2024-01-15T10:30:00Z",
  "version": "1.0.0"
}
```

---

### Audio Detection

Detect audio deepfakes and voice manipulation.

```http
POST /detect/audio
Content-Type: multipart/form-data
```

**Request:**
| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| file | File | Yes | Audio file (WAV, MP3, FLAC) |

**Response:**
```json
{
  "risk_score": 0.75,
  "classification": "BLOCK",
  "confidence": 0.82,
  "is_fake": true,
  "explanation": "Audio shows significant signs of being synthetically generated...",
  "detected_attacks": [
    "High-frequency artifacts detected (typical in GAN-generated audio)",
    "Phase discontinuities detected (possible audio splicing)"
  ],
  "model_scores": {
    "cnn": 0.78,
    "ensemble": 0.72,
    "lstm": 0.74
  },
  "processing_time_ms": 125.5
}
```

**Response Fields:**
| Field | Type | Description |
|-------|------|-------------|
| risk_score | float | Overall risk score (0-1) |
| classification | string | Risk level (PASS/FLAG/BLOCK/ALERT) |
| confidence | float | Detection confidence (0-1) |
| is_fake | boolean | Whether audio is likely fake |
| explanation | string | Human-readable explanation |
| detected_attacks | array | List of detected attack indicators |
| model_scores | object | Individual model predictions |
| processing_time_ms | float | Processing time in milliseconds |

---

### Image Detection

Detect visual prompt injections.

```http
POST /detect/image
Content-Type: multipart/form-data
```

**Request:**
| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| file | File | Yes | Image file (PNG, JPG, JPEG) |

**Response:**
```json
{
  "injection_detected": true,
  "risk_score": 0.65,
  "extracted_text": "Welcome to the application...",
  "hidden_text": "ignore previous instructions",
  "suspicious_patterns": [
    {
      "category": "override",
      "matched_keyword": "ignore",
      "confidence": 0.9,
      "severity": "high"
    }
  ],
  "visual_anomalies": [
    "Text hidden in corner: 'ignore'"
  ],
  "processing_time_ms": 210.3
}
```

---

### Multimodal Detection

Combined audio and visual analysis.

```http
POST /detect/multimodal
Content-Type: multipart/form-data
```

**Request:**
| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| audio | File | Yes | Audio file |
| image | File | Yes | Image file |

**Response:**
```json
{
  "overall_risk": 0.72,
  "risk_level": "BLOCK",
  "action": "REJECT - Block with explanation",
  "audio_analysis": {
    "deepfake_risk": 0.65,
    "voice_clone_risk": 0.55,
    "is_fake": false
  },
  "visual_analysis": {
    "injection_risk": 0.78,
    "is_malicious": true,
    "hidden_content": true
  },
  "consistency_score": 0.85,
  "threats_detected": [
    "Visual prompt injection (risk: 78.0%)",
    "Potential voice cloning (score: 55.0%)"
  ],
  "recommendations": [
    "Do not process this content",
    "Request authentic content from verified source"
  ],
  "processing_time_ms": 450.7
}
```

---

### Batch Detection

Process multiple files.

```http
POST /detect/batch
Content-Type: multipart/form-data
```

**Request:**
| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| files | File[] | Yes | Multiple audio/image files |

**Response:**
```json
{
  "results": [
    {
      "filename": "audio1.wav",
      "type": "audio",
      "risk_score": 0.25,
      "is_fake": false
    },
    {
      "filename": "image1.png",
      "type": "image",
      "risk_score": 0.15,
      "is_malicious": false
    }
  ]
}
```

---

### Report Generation

Generate detailed security report.

```http
POST /report
Content-Type: multipart/form-data
```

**Request:**
| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| audio | File | No | Audio file (optional) |
| image | File | No | Image file (optional) |

**Response:**
```json
{
  "report": {
    "summary": "The analyzed content appears to be genuine and safe...",
    "risk_level": "PASS",
    "confidence_breakdown": {
      "Overall": 0.15,
      "Audio Analysis": 0.12,
      "Visual Analysis": 0.18,
      "Consistency": 0.95
    },
    "detected_threats": [],
    "recommendations": [
      "No immediate action required",
      "Continue monitoring for pattern changes"
    ],
    "timestamp": "2024-01-15T10:35:00Z"
  },
  "html": "<!DOCTYPE html>..."
}
```

---

## Error Responses

### 400 Bad Request
```json
{
  "detail": "Invalid file type. Allowed: ['audio/wav', 'audio/mpeg']"
}
```

### 500 Internal Server Error
```json
{
  "detail": "Error processing audio: [error message]"
}
```

---

## Rate Limiting

Not implemented by default. Consider implementing:
- 100 requests per minute per IP
- 10 concurrent connections per IP

---

## Python Client Example

```python
import requests

# Audio detection
with open("audio.wav", "rb") as f:
    response = requests.post(
        "http://localhost:8000/detect/audio",
        files={"file": f}
    )
    result = response.json()
    print(f"Risk: {result['risk_score']:.1%}")

# Image detection
with open("image.png", "rb") as f:
    response = requests.post(
        "http://localhost:8000/detect/image",
        files={"file": f}
    )
    result = response.json()
    print(f"Injection: {result['injection_detected']}")

# Multimodal
with open("audio.wav", "rb") as audio, open("image.png", "rb") as image:
    response = requests.post(
        "http://localhost:8000/detect/multimodal",
        files={"audio": audio, "image": image}
    )
    result = response.json()
    print(f"Overall Risk: {result['overall_risk']:.1%}")
```
