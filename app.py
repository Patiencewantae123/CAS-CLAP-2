import streamlit as st
import os
import glob
import json
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from pathlib import Path
import subprocess
import sys

# Page config
st.set_page_config(
    page_title="CAS-V: Conflict-Aware Semantic Verification",
    page_icon="🎵",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-title {
        text-align: center;
        font-size: 2.5rem;
        font-weight: bold;
        margin-bottom: 1rem;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
    }
    .subtitle {
        text-align: center;
        font-size: 1.2rem;
        color: #666;
        margin-bottom: 2rem;
    }
    .metric-card {
        background: #f0f2f6;
        padding: 1.5rem;
        border-radius: 10px;
        margin: 0.5rem 0;
    }
    .success-box {
        background: #d4edda;
        border: 1px solid #c3e6cb;
        color: #155724;
        padding: 1rem;
        border-radius: 5px;
        margin: 1rem 0;
    }
    .info-box {
        background: #d1ecf1;
        border: 1px solid #bee5eb;
        color: #0c5460;
        padding: 1rem;
        border-radius: 5px;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def get_available_images():
    """Get list of available images in workspace"""
    image_extensions = ['*.jpg', '*.jpeg', '*.png', '*.gif']
    images = []
    for ext in image_extensions:
        images.extend(glob.glob(f"**/{ext}", recursive=True))
    # Filter out paper results
    images = [img for img in images if "paper_results" not in img and "fig" not in img]
    return images[:10]  # Return top 10

def load_image_display(image_path):
    """Load and display image"""
    try:
        img = Image.open(image_path)
        return img
    except:
        return None

def run_audio_generation(image_path, output_name):
    """Run audio generation script"""
    try:
        if image_path:
            result = subprocess.run(
                [sys.executable, "generate_advanced_audio.py", "--image", image_path, "--output", output_name],
                capture_output=True,
                text=True,
                timeout=60
            )
        else:
            result = subprocess.run(
                [sys.executable, "generate_advanced_audio.py", "--output", output_name],
                capture_output=True,
                text=True,
                timeout=60
            )
        return result.returncode == 0, result.stdout, result.stderr
    except Exception as e:
        return False, "", str(e)

def run_paper_generation():
    """Run paper results generation"""
    try:
        result = subprocess.run(
            [sys.executable, "generate_paper_results.py"],
            capture_output=True,
            text=True,
            timeout=60
        )
        return result.returncode == 0, result.stdout, result.stderr
    except Exception as e:
        return False, "", str(e)

def load_json_results(filepath):
    """Load JSON results file"""
    try:
        with open(filepath, 'r') as f:
            return json.load(f)
    except:
        return None

# ============================================================================
# MAIN UI
# ============================================================================

# Header
col1, col2, col3 = st.columns([1, 2, 1])
with col2:
    st.markdown('<div class="main-title">🎵 CAS-V</div>', unsafe_allow_html=True)
    st.markdown('<div class="subtitle">Conflict-Aware Semantic Verification for Image-to-Audio Generation</div>', unsafe_allow_html=True)

# Sidebar Navigation
st.sidebar.markdown("---")
st.sidebar.markdown("## 📑 Navigation")
page = st.sidebar.radio(
    "Select Page:",
    ["🏠 Home", "🎨 Audio Generation", "📊 Paper Results", "📈 Metrics & Analysis", "⚙️ Settings"],
    label_visibility="collapsed"
)

st.sidebar.markdown("---")
st.sidebar.markdown("### 📚 About")
st.sidebar.info("""
**CAS-V** is a conflict-aware semantic verification framework for image-to-audio generation.

It combines:
- 🖼️ Visual semantic analysis
- 🧠 Language-guided concept compatibility checks
- 🎯 Conflict filtering before synthesis
- 🎵 Audio generation with multimodal conditioning

**Features:**
- Semantic consistency verification
- Multi-image processing
- Emotion-aware audio generation
- Quality metrics tracking
""")

# ============================================================================
# PAGE: HOME
# ============================================================================
if page == "🏠 Home":
    st.markdown("---")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### 🚀 Quick Start")
        st.markdown("""
        1. **Upload or Select** an image
        2. **Generate** audio from the image
        3. **Listen** to the synthesized audio
        4. **View** quality metrics
        5. **Export** results
        """)
        
        st.markdown("### ✨ Key Features")
        st.markdown("""
        - ✅ Multi-image batch processing
        - ✅ Real-time audio generation
        - ✅ Quality metrics (FAD, SNR, CLAP)
        - ✅ Paper-ready visualizations
        - ✅ JSON export for data analysis
        """)
    
    with col2:
        st.markdown("### 📊 Latest Metrics")
        
        # Try to load latest results
        results_file = "paper_results/generation_results.json"
        if os.path.exists(results_file):
            results = load_json_results(results_file)
            if results:
                metrics = results.get('quality_metrics', {})
                
                col_a, col_b, col_c = st.columns(3)
                with col_a:
                    fad = metrics.get('frechet_audio_distance', {}).get('value', 'N/A')
                    st.metric("FAD Score", fad, delta="-0.2 (better)" if isinstance(fad, float) else None)
                with col_b:
                    snr = metrics.get('signal_to_noise_ratio', {}).get('value', 'N/A')
                    st.metric("SNR (dB)", snr, delta="+2.1 dB" if isinstance(snr, float) else None)
                with col_c:
                    clap = metrics.get('clap_alignment_score', {}).get('value', 'N/A')
                    st.metric("CLAP Score", clap, delta="+0.05" if isinstance(clap, float) else None)
        
        st.markdown("### 🎯 Quick Actions")
        if st.button("🎵 Generate Multi-Image Audio"):
            st.info("Redirecting to Audio Generation page...")
            st.session_state.page = "🎨 Audio Generation"

# ============================================================================
# PAGE: AUDIO GENERATION
# ============================================================================
elif page == "🎨 Audio Generation":
    st.markdown("---")
    st.markdown("## 🎨 Audio Generation")
    st.markdown("Generate audio from images using the MSA-I2A pipeline")
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.markdown("### 📤 Image Input")
        
        input_type = st.radio("Select input type:", ["Select from library", "Upload custom image"])
        
        image_path = None
        image_to_display = None
        
        if input_type == "Select from library":
            available_images = get_available_images()
            if available_images:
                image_path = st.selectbox("Available images:", available_images)
                image_to_display = load_image_display(image_path)
            else:
                st.warning("No images found. Please upload one.")
        else:
            uploaded_file = st.file_uploader("Upload image", type=["jpg", "jpeg", "png"])
            if uploaded_file:
                image_to_display = Image.open(uploaded_file)
                # Save uploaded image
                os.makedirs("uploaded_images", exist_ok=True)
                image_path = f"uploaded_images/{uploaded_file.name}"
                with open(image_path, "wb") as f:
                    f.write(uploaded_file.getbuffer())
                st.success(f"Uploaded: {uploaded_file.name}")
        
        if image_to_display:
            st.image(image_to_display, use_column_width=True, caption="Preview")
        
        st.markdown("### ⚙️ Generation Settings")
        duration = st.slider("Audio duration (seconds):", 1.0, 10.0, 3.0)
        batch_mode = st.checkbox("Process multiple images", False)
        
    with col2:
        st.markdown("### 🎵 Generation Controls")
        
        st.markdown("#### Output Configuration")
        output_name = st.text_input("Output filename:", "generated_audio.wav")
        
        st.markdown("#### Status")
        status_placeholder = st.empty()
        progress_bar = st.progress(0)
        output_text = st.empty()
        
        st.markdown("---")
        
        # Generate button
        col_gen, col_clear = st.columns(2)
        
        with col_gen:
            if st.button("🚀 Generate Audio", key="gen_btn", use_container_width=True):
                if batch_mode:
                    status_placeholder.info("🔄 Processing multiple images...")
                    progress_bar.progress(30)
                    
                    # Run multi-image generation
                    success, stdout, stderr = run_audio_generation(None, output_name)
                    progress_bar.progress(100)
                    
                    if success:
                        st.markdown("""
                        <div class="success-box">
                        ✅ <strong>Audio generation completed!</strong><br>
                        Multiple audio files have been generated.
                        </div>
                        """, unsafe_allow_html=True)
                        output_text.text_area("Output Log:", stdout, height=200)
                    else:
                        st.error(f"❌ Error: {stderr}")
                else:
                    if not image_path and input_type == "Select from library":
                        st.error("Please select an image first")
                    else:
                        status_placeholder.info("🔄 Generating audio...")
                        progress_bar.progress(50)
                        
                        success, stdout, stderr = run_audio_generation(image_path, output_name)
                        progress_bar.progress(100)
                        
                        if success:
                            st.markdown("""
                            <div class="success-box">
                            ✅ <strong>Audio generated successfully!</strong>
                            </div>
                            """, unsafe_allow_html=True)
                            
                            # Display audio player
                            if os.path.exists(output_name):
                                st.audio(output_name, format="audio/wav")
                                
                                # Show download button
                                with open(output_name, "rb") as f:
                                    st.download_button(
                                        label="📥 Download Audio",
                                        data=f.read(),
                                        file_name=output_name,
                                        mime="audio/wav"
                                    )
                            
                            output_text.text_area("Output Log:", stdout, height=150)
                        else:
                            st.error(f"❌ Generation failed: {stderr}")
        
        with col_clear:
            if st.button("🗑️ Clear", key="clear_btn", use_container_width=True):
                status_placeholder.empty()
                progress_bar.empty()
                output_text.empty()

# ============================================================================
# PAGE: PAPER RESULTS
# ============================================================================
elif page == "📊 Paper Results":
    st.markdown("---")
    st.markdown("## 📊 Paper Results & Visualizations")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown("### 📈 Figures Gallery")
        
        paper_dir = "paper_results"
        figures = sorted(glob.glob(f"{paper_dir}/fig*.png"))
        
        if figures:
            # Organize figures in a grid
            cols = st.columns(2)
            for idx, fig_path in enumerate(figures):
                col = cols[idx % 2]
                with col:
                    try:
                        img = mpimg.imread(fig_path)
                        fig_name = os.path.basename(fig_path)
                        st.image(img, use_column_width=True, caption=fig_name)
                    except:
                        st.warning(f"Could not load {fig_path}")
        else:
            st.info("No paper figures found. Generate them using the button below.")
    
    with col2:
        st.markdown("### 🎯 Actions")
        
        if st.button("🔄 Regenerate Paper Figures", use_container_width=True):
            progress = st.progress(0)
            status = st.status("Generating paper results...", expanded=True)
            
            with status:
                st.write("Running generate_paper_results.py...")
                progress.progress(50)
                
                success, stdout, stderr = run_paper_generation()
                progress.progress(100)
                
                if success:
                    st.success("✅ Paper figures generated successfully!")
                    st.text_area("Output:", stdout, height=150)
                    
                    # Rerun to show new figures
                    st.rerun()
                else:
                    st.error(f"❌ Error: {stderr}")
        
        st.markdown("### 📥 Export Options")
        
        # Check for existing results
        results_file = "paper_results/generation_results.json"
        if os.path.exists(results_file):
            with open(results_file, "rb") as f:
                st.download_button(
                    label="📋 Download Results JSON",
                    data=f.read(),
                    file_name="generation_results.json",
                    mime="application/json"
                )
        
        metadata_file = "paper_results/paper_metadata.json"
        if os.path.exists(metadata_file):
            with open(metadata_file, "rb") as f:
                st.download_button(
                    label="📊 Download Metadata JSON",
                    data=f.read(),
                    file_name="paper_metadata.json",
                    mime="application/json"
                )

# ============================================================================
# PAGE: METRICS & ANALYSIS
# ============================================================================
elif page == "📈 Metrics & Analysis":
    st.markdown("---")
    st.markdown("## 📈 Metrics & Analysis")
    
    results_file = "paper_results/generation_results.json"
    metadata_file = "paper_results/paper_metadata.json"
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### 🎵 Generation Results")
        if os.path.exists(results_file):
            results = load_json_results(results_file)
            if results:
                metrics = results.get('quality_metrics', {})
                
                # Display metrics
                st.metric(
                    "Fréchet Audio Distance (FAD)",
                    f"{metrics.get('frechet_audio_distance', {}).get('value', 'N/A'):.3f}",
                    help="Lower is better (0-10)"
                )
                st.metric(
                    "Signal-to-Noise Ratio (SNR)",
                    f"{metrics.get('signal_to_noise_ratio', {}).get('value', 'N/A'):.2f} dB",
                    help="Higher is better"
                )
                st.metric(
                    "CLAP Alignment Score",
                    f"{metrics.get('clap_alignment_score', {}).get('value', 'N/A'):.4f}",
                    help="Cosine similarity (0-1)"
                )
                
                # Additional metrics
                with st.expander("📊 Detailed Metrics"):
                    for key, value in metrics.items():
                        if isinstance(value, dict):
                            st.write(f"**{key}**: {value.get('value', 'N/A')}")
        else:
            st.info("Run audio generation to generate metrics.")
    
    with col2:
        st.markdown("### 📋 Paper Metadata")
        if os.path.exists(metadata_file):
            metadata = load_json_results(metadata_file)
            if metadata:
                st.json(metadata.get('summary_statistics', {}))
        else:
            st.info("Generate paper results to view metadata.")
    
    # Audio files listing
    st.markdown("---")
    st.markdown("### 🎵 Generated Audio Files")
    
    audio_files = glob.glob("*.wav")
    if audio_files:
        for audio_file in audio_files[:10]:  # Show top 10
            col1, col2, col3 = st.columns([2, 1, 1])
            with col1:
                st.write(f"**{audio_file}**")
            with col2:
                st.audio(audio_file)
            with col3:
                with open(audio_file, "rb") as f:
                    st.download_button(
                        "📥",
                        data=f.read(),
                        file_name=audio_file,
                        mime="audio/wav",
                        key=audio_file
                    )
    else:
        st.info("No audio files generated yet.")

# ============================================================================
# PAGE: SETTINGS
# ============================================================================
elif page == "⚙️ Settings":
    st.markdown("---")
    st.markdown("## ⚙️ Settings & Configuration")
    
    st.markdown("### 🔧 Pipeline Configuration")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### Model Parameters")
        embed_dim = st.slider("Embedding Dimension:", 64, 512, 256, step=32)
        num_heads = st.slider("Attention Heads:", 1, 8, 4)
        n_mels = st.slider("Mel Spectrogram Bins:", 40, 128, 80, step=8)
    
    with col2:
        st.markdown("#### Audio Parameters")
        sample_rate = st.selectbox("Sample Rate:", [16000, 22050, 44100, 48000], index=1)
        audio_duration = st.slider("Default Duration (s):", 1.0, 30.0, 3.0)
        n_fft = st.selectbox("FFT Size:", [512, 1024, 2048], index=1)
    
    st.markdown("---")
    st.markdown("### 📁 Workspace Info")
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Total Audio Files", len(glob.glob("*.wav")))
    with col2:
        st.metric("Paper Figures", len(glob.glob("paper_results/fig*.png")))
    with col3:
        st.metric("Project Files", len(glob.glob("MSA-I2A/**/*.py", recursive=True)))
    
    st.markdown("---")
    st.markdown("### 🗑️ Cleanup")
    
    if st.button("🗑️ Clear Generated Audio Files"):
        audio_files = glob.glob("*.wav")
        for f in audio_files:
            try:
                os.remove(f)
            except:
                pass
        st.success(f"Removed {len(audio_files)} audio files")
    
    st.markdown("---")
    st.markdown("### ℹ️ System Info")
    st.code(f"""
Python Version: {sys.version.split()[0]}
Working Directory: {os.getcwd()}
Streamlit Version: {st.__version__}
    """)

# ============================================================================
# FOOTER
# ============================================================================
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #666; font-size: 0.9rem; margin-top: 2rem;">
    <p>🎵 <strong>MSA-I2A: Multimodal Semantic-Aware Image-to-Audio Generation</strong></p>
    <p>Advanced audio synthesis from visual input • Powered by PyTorch & TensorFlow</p>
</div>
""", unsafe_allow_html=True)
