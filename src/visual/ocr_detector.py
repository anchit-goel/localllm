"""
OCR-based Text Detection in Images.

Extracts and analyzes text from images for prompt injection detection:
- Primary text extraction using Tesseract/EasyOCR
- Hidden text detection (low contrast, small fonts)
- Position-based anomaly detection
"""

import numpy as np
from typing import Dict, Optional, List, Tuple, Union
from dataclasses import dataclass
import logging
from pathlib import Path

try:
    import cv2
except ImportError:
    cv2 = None

try:
    from PIL import Image, ImageEnhance, ImageFilter
except ImportError:
    Image = None

logger = logging.getLogger(__name__)


@dataclass
class TextRegion:
    """Container for detected text region."""
    text: str
    confidence: float
    bbox: Tuple[int, int, int, int]  # x, y, width, height
    is_hidden: bool
    hiding_method: Optional[str]
    font_size_estimate: float
    contrast_ratio: float
    
    def to_dict(self) -> Dict:
        return {
            'text': self.text,
            'confidence': self.confidence,
            'bbox': self.bbox,
            'is_hidden': self.is_hidden,
            'hiding_method': self.hiding_method,
            'font_size_estimate': self.font_size_estimate,
            'contrast_ratio': self.contrast_ratio,
        }


@dataclass
class OCRResult:
    """Container for OCR detection results."""
    full_text: str
    text_regions: List[TextRegion]
    hidden_text_detected: bool
    hidden_text: str
    total_confidence: float
    anomalies: List[str]
    
    def to_dict(self) -> Dict:
        return {
            'full_text': self.full_text,
            'text_regions': [r.to_dict() for r in self.text_regions],
            'hidden_text_detected': self.hidden_text_detected,
            'hidden_text': self.hidden_text,
            'total_confidence': self.total_confidence,
            'anomalies': self.anomalies,
        }


class OCRDetector:
    """
    Multi-engine OCR detector for text extraction and analysis.
    
    Supports:
    - Tesseract OCR
    - EasyOCR
    - Custom preprocessing for hidden text detection
    """
    
    def __init__(
        self,
        use_tesseract: bool = True,
        use_easyocr: bool = True,
        languages: List[str] = ['en'],
    ):
        """
        Initialize OCR detector.
        
        Args:
            use_tesseract: Enable Tesseract OCR
            use_easyocr: Enable EasyOCR
            languages: List of language codes
        """
        self.use_tesseract = use_tesseract
        self.use_easyocr = use_easyocr
        self.languages = languages
        
        self.tesseract_available = False
        self.easyocr_available = False
        self.easyocr_reader = None
        
        self._initialize_engines()
    
    def _initialize_engines(self):
        """Initialize OCR engines."""
        # Try Tesseract
        if self.use_tesseract:
            try:
                import pytesseract
                pytesseract.get_tesseract_version()
                self.tesseract_available = True
                logger.info("Tesseract OCR initialized")
            except Exception as e:
                logger.warning(f"Tesseract not available: {e}")
        
        # Try EasyOCR
        if self.use_easyocr:
            try:
                import easyocr
                self.easyocr_reader = easyocr.Reader(self.languages, gpu=False)
                self.easyocr_available = True
                logger.info("EasyOCR initialized")
            except Exception as e:
                logger.warning(f"EasyOCR not available: {e}")
    
    def load_image(self, image_input: Union[str, bytes, np.ndarray]) -> np.ndarray:
        """
        Load image from various sources.
        
        Args:
            image_input: Path, bytes, or numpy array
            
        Returns:
            Image as numpy array (BGR format for OpenCV)
        """
        if cv2 is None or Image is None:
            raise ImportError("OpenCV and Pillow are required for image processing")
        
        if isinstance(image_input, str):
            image = cv2.imread(image_input)
            if image is None:
                raise ValueError(f"Could not load image from {image_input}")
        elif isinstance(image_input, bytes):
            nparr = np.frombuffer(image_input, np.uint8)
            image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        elif isinstance(image_input, np.ndarray):
            image = image_input
        else:
            # Try PIL Image
            if hasattr(image_input, 'convert'):
                image = np.array(image_input.convert('RGB'))
                image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            else:
                raise ValueError(f"Unsupported image input type: {type(image_input)}")
        
        return image
    
    def extract_text_tesseract(
        self,
        image: np.ndarray,
        config: str = '--oem 3 --psm 6',
    ) -> Tuple[str, List[Dict]]:
        """
        Extract text using Tesseract.
        
        Args:
            image: Image as numpy array
            config: Tesseract config string
            
        Returns:
            Tuple of (full_text, list of word details)
        """
        if not self.tesseract_available:
            return "", []
        
        import pytesseract
        
        # Convert to RGB for Tesseract
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Get full text
        full_text = pytesseract.image_to_string(rgb_image, config=config)
        
        # Get detailed data
        data = pytesseract.image_to_data(rgb_image, output_type=pytesseract.Output.DICT, config=config)
        
        words = []
        for i in range(len(data['text'])):
            text = data['text'][i].strip()
            if text:
                words.append({
                    'text': text,
                    'confidence': data['conf'][i],
                    'x': data['left'][i],
                    'y': data['top'][i],
                    'width': data['width'][i],
                    'height': data['height'][i],
                })
        
        return full_text, words
    
    def extract_text_easyocr(self, image: np.ndarray) -> Tuple[str, List[Dict]]:
        """
        Extract text using EasyOCR.
        
        Args:
            image: Image as numpy array
            
        Returns:
            Tuple of (full_text, list of word details)
        """
        if not self.easyocr_available or self.easyocr_reader is None:
            return "", []
        
        # EasyOCR expects RGB
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        results = self.easyocr_reader.readtext(rgb_image)
        
        words = []
        texts = []
        for (bbox, text, conf) in results:
            texts.append(text)
            # Convert bbox to x, y, w, h
            x_coords = [p[0] for p in bbox]
            y_coords = [p[1] for p in bbox]
            x = min(x_coords)
            y = min(y_coords)
            w = max(x_coords) - x
            h = max(y_coords) - y
            
            words.append({
                'text': text,
                'confidence': conf * 100,  # Scale to match Tesseract
                'x': int(x),
                'y': int(y),
                'width': int(w),
                'height': int(h),
            })
        
        full_text = ' '.join(texts)
        return full_text, words
    
    def detect_hidden_text(
        self,
        image: np.ndarray,
    ) -> List[TextRegion]:
        """
        Detect hidden text using various preprocessing techniques.
        
        Detects:
        - Low contrast text
        - Very small fonts
        - Text matching background color
        - Transparent text overlays
        
        Args:
            image: Image as numpy array
            
        Returns:
            List of detected hidden text regions
        """
        hidden_regions = []
        
        # 1. High contrast enhancement
        enhanced = self._enhance_contrast(image)
        enhanced_text, enhanced_words = self._extract_text_combined(enhanced)
        
        # 2. Inverted image
        inverted = cv2.bitwise_not(image)
        inverted_text, inverted_words = self._extract_text_combined(inverted)
        
        # 3. Edge detection for very faint text
        edges = self._extract_edges(image)
        edge_text, edge_words = self._extract_text_combined(edges)
        
        # 4. Color channel separation
        for i, channel_name in enumerate(['Blue', 'Green', 'Red']):
            channel = image[:, :, i]
            channel_3d = cv2.merge([channel, channel, channel])
            channel_text, channel_words = self._extract_text_combined(channel_3d)
            
            # Check for text only visible in specific channels
            for word in channel_words:
                if word['text'] and word['confidence'] > 50:
                    # Calculate local contrast
                    contrast = self._calculate_local_contrast(
                        image, word['x'], word['y'], word['width'], word['height']
                    )
                    
                    if contrast < 0.3:  # Low contrast indicates hidden text
                        hidden_regions.append(TextRegion(
                            text=word['text'],
                            confidence=word['confidence'],
                            bbox=(word['x'], word['y'], word['width'], word['height']),
                            is_hidden=True,
                            hiding_method=f'color_channel_{channel_name.lower()}',
                            font_size_estimate=word['height'],
                            contrast_ratio=contrast,
                        ))
        
        # 5. Saturation reduction (for colored hidden text)
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        hsv[:, :, 1] = 0  # Remove saturation
        gray_rgb = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
        sat_text, sat_words = self._extract_text_combined(gray_rgb)
        
        # Compare with original OCR to find new detections
        original_text, original_words = self._extract_text_combined(image)
        original_texts = set(w['text'].lower() for w in original_words if w['text'])
        
        for word in enhanced_words + inverted_words + edge_words + sat_words:
            if word['text'] and word['text'].lower() not in original_texts:
                contrast = self._calculate_local_contrast(
                    image, word['x'], word['y'], word['width'], word['height']
                )
                
                hidden_regions.append(TextRegion(
                    text=word['text'],
                    confidence=word['confidence'],
                    bbox=(word['x'], word['y'], word['width'], word['height']),
                    is_hidden=True,
                    hiding_method='low_contrast',
                    font_size_estimate=word['height'],
                    contrast_ratio=contrast,
                ))
        
        # 6. Detect very small text (might be intentionally hidden)
        for word in original_words:
            if word['height'] < 10 and word['text']:
                hidden_regions.append(TextRegion(
                    text=word['text'],
                    confidence=word['confidence'],
                    bbox=(word['x'], word['y'], word['width'], word['height']),
                    is_hidden=True,
                    hiding_method='small_font',
                    font_size_estimate=word['height'],
                    contrast_ratio=1.0,
                ))
        
        return hidden_regions
    
    def _enhance_contrast(self, image: np.ndarray) -> np.ndarray:
        """Apply aggressive contrast enhancement."""
        # Convert to LAB color space
        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        l_channel = lab[:, :, 0]
        
        # Apply CLAHE
        clahe = cv2.createCLAHE(clipLimit=4.0, tileGridSize=(8, 8))
        enhanced_l = clahe.apply(l_channel)
        
        lab[:, :, 0] = enhanced_l
        enhanced = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
        
        return enhanced
    
    def _extract_edges(self, image: np.ndarray) -> np.ndarray:
        """Extract edges for faint text detection."""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 50, 150)
        return cv2.merge([edges, edges, edges])
    
    def _extract_text_combined(self, image: np.ndarray) -> Tuple[str, List[Dict]]:
        """Extract text using available engines."""
        all_words = []
        texts = []
        
        if self.tesseract_available:
            text, words = self.extract_text_tesseract(image)
            texts.append(text)
            all_words.extend(words)
        
        if self.easyocr_available:
            text, words = self.extract_text_easyocr(image)
            texts.append(text)
            all_words.extend(words)
        
        combined_text = ' '.join(texts)
        return combined_text, all_words
    
    def _calculate_local_contrast(
        self,
        image: np.ndarray,
        x: int,
        y: int,
        width: int,
        height: int,
    ) -> float:
        """Calculate contrast ratio in region."""
        # Extract region with padding
        pad = 5
        y1 = max(0, y - pad)
        y2 = min(image.shape[0], y + height + pad)
        x1 = max(0, x - pad)
        x2 = min(image.shape[1], x + width + pad)
        
        region = image[y1:y2, x1:x2]
        
        if region.size == 0:
            return 1.0
        
        gray = cv2.cvtColor(region, cv2.COLOR_BGR2GRAY)
        
        # Calculate contrast as std of pixel values
        contrast = np.std(gray) / 127.5
        return min(1.0, contrast)
    
    def detect_position_anomalies(
        self,
        image: np.ndarray,
        text_regions: List[Dict],
    ) -> List[str]:
        """
        Detect text in unusual positions.
        
        Args:
            image: Image as numpy array
            text_regions: List of detected text regions
            
        Returns:
            List of anomaly descriptions
        """
        anomalies = []
        height, width = image.shape[:2]
        
        for region in text_regions:
            x, y = region['x'], region['y']
            w, h = region['width'], region['height']
            
            # Check for text at very edges
            if x < 5 or y < 5:
                anomalies.append(f"Text at image edge: '{region['text']}'")
            
            if x + w > width - 5 or y + h > height - 5:
                anomalies.append(f"Text at image boundary: '{region['text']}'")
            
            # Check for text in corners (common hiding spot)
            corner_threshold = min(width, height) * 0.05
            if (x < corner_threshold and y < corner_threshold) or \
               (x + w > width - corner_threshold and y < corner_threshold) or \
               (x < corner_threshold and y + h > height - corner_threshold) or \
               (x + w > width - corner_threshold and y + h > height - corner_threshold):
                anomalies.append(f"Text hidden in corner: '{region['text']}'")
            
            # Very small text
            if h < 8:
                anomalies.append(f"Very small text detected: '{region['text']}'")
        
        return anomalies
    
    def detect(
        self,
        image_input: Union[str, bytes, np.ndarray],
    ) -> OCRResult:
        """
        Full OCR detection pipeline.
        
        Args:
            image_input: Image to analyze
            
        Returns:
            OCRResult with all detected text and analysis
        """
        # Load image
        image = self.load_image(image_input)
        
        # Primary OCR
        full_text, primary_words = self._extract_text_combined(image)
        
        # Detect hidden text
        hidden_regions = self.detect_hidden_text(image)
        
        # Convert primary words to TextRegions
        text_regions = []
        for word in primary_words:
            if word['text']:
                contrast = self._calculate_local_contrast(
                    image, word['x'], word['y'], word['width'], word['height']
                )
                text_regions.append(TextRegion(
                    text=word['text'],
                    confidence=word['confidence'],
                    bbox=(word['x'], word['y'], word['width'], word['height']),
                    is_hidden=False,
                    hiding_method=None,
                    font_size_estimate=word['height'],
                    contrast_ratio=contrast,
                ))
        
        # Add hidden regions
        text_regions.extend(hidden_regions)
        
        # Detect position anomalies
        anomalies = self.detect_position_anomalies(image, primary_words)
        
        # Collect hidden text
        hidden_text_parts = [r.text for r in hidden_regions if r.text]
        hidden_text = ' '.join(hidden_text_parts)
        
        # Calculate total confidence
        confidences = [r.confidence for r in text_regions if r.confidence > 0]
        total_confidence = np.mean(confidences) if confidences else 0.0
        
        return OCRResult(
            full_text=full_text,
            text_regions=text_regions,
            hidden_text_detected=len(hidden_regions) > 0,
            hidden_text=hidden_text,
            total_confidence=float(total_confidence),
            anomalies=anomalies,
        )
    
    def get_annotated_image(
        self,
        image_input: Union[str, bytes, np.ndarray],
        ocr_result: OCRResult,
    ) -> np.ndarray:
        """
        Generate image with text regions annotated.
        
        Args:
            image_input: Original image
            ocr_result: OCR detection result
            
        Returns:
            Annotated image as numpy array
        """
        image = self.load_image(image_input)
        annotated = image.copy()
        
        for region in ocr_result.text_regions:
            x, y, w, h = region.bbox
            
            # Color based on hidden status
            if region.is_hidden:
                color = (0, 0, 255)  # Red for hidden
            else:
                color = (0, 255, 0)  # Green for normal
            
            cv2.rectangle(annotated, (x, y), (x + w, y + h), color, 2)
            
            # Add label
            label = region.text[:20]
            cv2.putText(annotated, label, (x, y - 5),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
        
        return annotated
