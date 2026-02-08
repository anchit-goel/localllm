"""
Streamlit Web Interface for Multimodal Security System.

Interactive demo for detecting:
- Audio deepfakes
- Voice cloning
- Visual prompt injections
- Cross-modal attacks
"""

import streamlit as st
import numpy as np
import os
import sys
import io
import tempfile
from datetime import datetime

# Add src to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Page configuration
st.set_page_config(
    page_title="Multimodal AI Security System",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        margin-bottom: 2rem;
    }
    .risk-high { color: #e74c3c; font-weight: bold; }
    .risk-medium { color: #f39c12; font-weight: bold; }
    .risk-low { color: #27ae60; font-weight: bold; }
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1rem;
        border-radius: 10px;
        color: white;
        text-align: center;
    }
    .threat-alert {
        background: #ffe6e6;
        border-left: 4px solid #e74c3c;
        padding: 1rem;
        margin: 0.5rem 0;
    }
    .safe-alert {
        background: #e6ffe6;
        border-left: 4px solid #27ae60;
        padding: 1rem;
        margin: 0.5rem 0;
    }
</style>
""", unsafe_allow_html=True)


# Initialize session state
if 'audio_result' not in st.session_state:
    st.session_state.audio_result = None
if 'visual_result' not in st.session_state:
    st.session_state.visual_result = None


@st.cache_resource
def load_detectors():
    """Load detection models (cached)."""
    try:
        from src.audio.deepfake_detector import DeepfakeDetector
        from src.audio.voice_clone_detector import VoiceCloneDetector
        from src.visual.injection_analyzer import InjectionAnalyzer
        from src.visual.steganography_checker import SteganographyChecker
        from src.scoring.risk_engine import RiskEngine
        from src.scoring.explainer import Explainer
        
        return {
            'deepfake': DeepfakeDetector(),
            'voice_clone': VoiceCloneDetector(),
            'injection': InjectionAnalyzer(),
            'steganography': SteganographyChecker(),
            'risk_engine': RiskEngine(),
            'explainer': Explainer(),
        }
    except Exception as e:
        st.error(f"Error loading models: {e}")
        return None


def get_risk_color(score):
    """Get color based on risk score."""
    if score < 0.3:
        return "green"
    elif score < 0.6:
        return "orange"
    else:
        return "red"


def display_risk_meter(score, title="Risk Score"):
    """Display risk score as a progress bar with color."""
    color = get_risk_color(score)
    st.markdown(f"### {title}")
    st.progress(score)
    st.markdown(f"<h2 style='color: {color};'>{score:.1%}</h2>", unsafe_allow_html=True)


def main():
    # Header
    st.markdown("<h1 class='main-header'>🛡️ Multimodal AI Security System</h1>", unsafe_allow_html=True)
    st.markdown("---")
    
    # Sidebar
    with st.sidebar:
        st.image("https://img.icons8.com/color/96/000000/security-checked.png", width=80)
        st.title("Navigation")
        
        page = st.radio(
            "Select Analysis Type:",
            ["🎵 Audio Analysis", "🖼️ Image Analysis", "🔄 Multimodal Analysis", "📊 Dashboard"]
        )
        
        st.markdown("---")
        st.markdown("### About")
        st.markdown("""
        This system detects:
        - 🎤 Audio deepfakes
        - 👤 Voice cloning
        - 📝 Visual prompt injections
        - 🔗 Cross-modal attacks
        """)
        
        st.markdown("---")
        st.markdown("### Settings")
        sensitivity = st.slider("Detection Sensitivity", 0.0, 1.0, 0.5)
        show_details = st.checkbox("Show Technical Details", value=True)
    
    # Load detectors
    detectors = load_detectors()
    if detectors is None:
        st.error("Failed to load detection models. Please check the installation.")
        return
    
    # Audio Analysis Page
    if page == "🎵 Audio Analysis":
        st.header("Audio Deepfake & Voice Clone Detection")
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            audio_file = st.file_uploader(
                "Upload Audio File",
                type=['wav', 'mp3', 'flac'],
                help="Supported formats: WAV, MP3, FLAC"
            )
            
            if audio_file:
                st.audio(audio_file)
                
                if st.button("🔍 Analyze Audio", type="primary"):
                    with st.spinner("Analyzing audio for deepfakes..."):
                        try:
                            # Save to temp file
                            with tempfile.NamedTemporaryFile(delete=False, suffix='.wav') as tmp:
                                tmp.write(audio_file.read())
                                tmp_path = tmp.name
                            
                            # Run detection
                            result = detectors['deepfake'].detect(tmp_path)
                            voice_result = detectors['voice_clone'].detect(tmp_path)
                            
                            st.session_state.audio_result = result
                            st.session_state.voice_result = voice_result
                            
                            os.unlink(tmp_path)
                            
                        except Exception as e:
                            st.error(f"Analysis failed: {e}")
        
        with col2:
            if st.session_state.audio_result:
                result = st.session_state.audio_result
                
                display_risk_meter(result.risk_score, "Deepfake Risk")
                
                st.markdown("### Classification")
                if result.is_fake:
                    st.markdown("<div class='threat-alert'>⚠️ <strong>POTENTIAL DEEPFAKE DETECTED</strong></div>", unsafe_allow_html=True)
                else:
                    st.markdown("<div class='safe-alert'>✅ <strong>Audio appears genuine</strong></div>", unsafe_allow_html=True)
        
        # Detailed results
        if st.session_state.audio_result and show_details:
            result = st.session_state.audio_result
            
            st.markdown("---")
            st.subheader("Detailed Analysis")
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("CNN Score", f"{result.cnn_score:.1%}", 
                         delta=f"{result.cnn_score - 0.5:.1%}")
            with col2:
                st.metric("Ensemble Score", f"{result.ensemble_score:.1%}",
                         delta=f"{result.ensemble_score - 0.5:.1%}")
            with col3:
                st.metric("LSTM Score", f"{result.lstm_score:.1%}",
                         delta=f"{result.lstm_score - 0.5:.1%}")
            
            st.markdown("### Explanation")
            st.info(result.explanation)
            
            if result.detected_artifacts:
                st.markdown("### Detected Artifacts")
                for artifact in result.detected_artifacts:
                    st.warning(f"⚠️ {artifact}")
    
    # Image Analysis Page
    elif page == "🖼️ Image Analysis":
        st.header("Visual Prompt Injection Detection")
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            image_file = st.file_uploader(
                "Upload Image",
                type=['png', 'jpg', 'jpeg'],
                help="Supported formats: PNG, JPG, JPEG"
            )
            
            if image_file:
                st.image(image_file, caption="Uploaded Image", use_column_width=True)
                
                if st.button("🔍 Analyze Image", type="primary"):
                    with st.spinner("Analyzing image for prompt injections..."):
                        try:
                            content = image_file.read()
                            
                            result = detectors['injection'].analyze(content)
                            steg_result = detectors['steganography'].detect(content)
                            
                            st.session_state.visual_result = result
                            st.session_state.steg_result = steg_result
                            
                        except Exception as e:
                            st.error(f"Analysis failed: {e}")
        
        with col2:
            if st.session_state.visual_result:
                result = st.session_state.visual_result
                
                display_risk_meter(result.risk_score, "Injection Risk")
                
                st.markdown("### Classification")
                if result.is_malicious:
                    st.markdown("<div class='threat-alert'>⚠️ <strong>INJECTION DETECTED</strong></div>", unsafe_allow_html=True)
                else:
                    st.markdown("<div class='safe-alert'>✅ <strong>Image appears safe</strong></div>", unsafe_allow_html=True)
        
        # Detailed results
        if st.session_state.visual_result and show_details:
            result = st.session_state.visual_result
            
            st.markdown("---")
            st.subheader("Extracted Text")
            
            if result.extracted_text:
                st.text_area("Visible Text", result.extracted_text[:1000], height=150)
            
            if result.hidden_text:
                st.error(f"🔴 Hidden Text Found: {result.hidden_text[:500]}")
            
            st.markdown("### Explanation")
            st.info(result.explanation)
            
            if result.indicators:
                st.markdown("### Suspicious Patterns")
                for ind in result.indicators[:5]:
                    st.warning(f"[{ind.severity.upper()}] {ind.category}: {ind.matched_keyword}")
    
    # Multimodal Analysis Page
    elif page == "🔄 Multimodal Analysis":
        st.header("Cross-Modal Security Analysis")
        
        st.markdown("Upload both audio and image for comprehensive multimodal analysis.")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("🎵 Audio Input")
            audio_file = st.file_uploader("Upload Audio", type=['wav', 'mp3'], key="mm_audio")
            if audio_file:
                st.audio(audio_file)
        
        with col2:
            st.subheader("🖼️ Image Input")
            image_file = st.file_uploader("Upload Image", type=['png', 'jpg'], key="mm_image")
            if image_file:
                st.image(image_file, use_column_width=True)
        
        if audio_file and image_file:
            if st.button("🔍 Run Multimodal Analysis", type="primary"):
                with st.spinner("Performing cross-modal analysis..."):
                    try:
                        # Save audio to temp
                        with tempfile.NamedTemporaryFile(delete=False, suffix='.wav') as tmp:
                            tmp.write(audio_file.read())
                            audio_path = tmp.name
                        
                        image_content = image_file.read()
                        
                        # Run all analyses
                        audio_result = detectors['deepfake'].detect(audio_path)
                        visual_result = detectors['injection'].analyze(image_content)
                        
                        # Risk assessment
                        assessment = detectors['risk_engine'].assess(
                            deepfake_result=audio_result.to_dict(),
                            injection_result=visual_result.to_dict(),
                        )
                        
                        os.unlink(audio_path)
                        
                        # Display results
                        st.markdown("---")
                        st.subheader("Overall Risk Assessment")
                        
                        col1, col2, col3 = st.columns(3)
                        
                        with col1:
                            display_risk_meter(assessment.overall_risk, "Overall Risk")
                        with col2:
                            display_risk_meter(assessment.audio_risk, "Audio Risk")
                        with col3:
                            display_risk_meter(assessment.visual_risk, "Visual Risk")
                        
                        # Risk level badge
                        level = assessment.risk_level.value
                        if level == "ALERT":
                            st.error(f"🔴 RISK LEVEL: {level}")
                        elif level == "BLOCK":
                            st.warning(f"🟠 RISK LEVEL: {level}")
                        elif level == "FLAG":
                            st.info(f"🟡 RISK LEVEL: {level}")
                        else:
                            st.success(f"🟢 RISK LEVEL: {level}")
                        
                        # Threats
                        if assessment.threats_detected:
                            st.markdown("### Detected Threats")
                            for threat in assessment.threats_detected:
                                st.error(f"⚠️ {threat}")
                        
                        # Recommendations
                        st.markdown("### Recommendations")
                        for rec in assessment.recommendations:
                            st.info(f"💡 {rec}")
                        
                    except Exception as e:
                        st.error(f"Analysis failed: {e}")
    
    # Dashboard Page
    elif page == "📊 Dashboard":
        st.header("Security Analytics Dashboard")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("System Status")
            st.success("✅ All detection modules loaded")
            st.info(f"🕐 Last updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
            
            st.subheader("Detection Capabilities")
            capabilities = {
                "Audio Deepfake Detection": "✅ Active",
                "Voice Cloning Detection": "✅ Active",
                "Visual Injection Analysis": "✅ Active",
                "Steganography Detection": "✅ Active",
                "Cross-Modal Consistency": "✅ Active",
            }
            for cap, status in capabilities.items():
                st.write(f"{cap}: {status}")
        
        with col2:
            st.subheader("Attack Types Covered")
            
            attacks = [
                ("🎤 Text-to-Speech Deepfakes", "Synthetic speech detection"),
                ("👤 Voice Cloning", "Cloned voice identification"),
                ("🔄 Replay Attacks", "Audio replay detection"),
                ("📝 Visual Prompt Injection", "Hidden text/commands"),
                ("🔗 Cross-Modal Attacks", "Multimodal inconsistency"),
            ]
            
            for attack, desc in attacks:
                with st.expander(attack):
                    st.write(desc)
        
        st.markdown("---")
        st.subheader("Quick Test")
        
        test_type = st.selectbox("Select Test Type", ["Audio Test", "Image Test"])
        
        if test_type == "Audio Test":
            st.info("Generate test audio using text-to-speech and upload to test detection.")
        else:
            st.info("Create an image with hidden text and upload to test injection detection.")


if __name__ == "__main__":
    main()
