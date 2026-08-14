import streamlit as st
import os
import glob
import json
import numpy as np
import pandas as pd
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from pathlib import Path
import subprocess
import sys
from scipy.io.wavfile import write as wav_write

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
    /* Main app styling */
    .stApp {
        background:
            radial-gradient(circle at top left, rgba(96, 165, 250, 0.16), transparent 30%),
            radial-gradient(circle at top right, rgba(168, 85, 247, 0.12), transparent 30%),
            #f8fafc;
        color: #111827;
    }
    .main .block-container {
        padding-top: 2rem;
        padding-bottom: 3rem;
        max-width: 1400px;
    }
    
    /* Header */
    .soft-header {
        background: linear-gradient(135deg, rgba(255,255,255,0.95), rgba(239,246,255,0.95));
        border: 1px solid rgba(148,163,184,0.3);
        border-radius: 24px;
        padding: 2rem 2.5rem;
        margin-bottom: 2rem;
        box-shadow: 0 20px 40px rgba(15, 23, 42, 0.08);
        animation: slideDown 0.5s ease;
    }
    @keyframes slideDown {
        from { opacity: 0; transform: translateY(-10px); }
        to { opacity: 1; transform: translateY(0); }
    }
    .soft-header .brand {
        display: inline-block;
        background: linear-gradient(135deg, #2563eb, #7c3aed, #0ea5e9);
        color: white;
        border-radius: 999px;
        padding: 0.4rem 0.9rem;
        font-size: 0.7rem;
        font-weight: 800;
        letter-spacing: 0.12em;
        text-transform: uppercase;
        margin-bottom: 0.8rem;
        box-shadow: 0 4px 12px rgba(37, 99, 235, 0.25);
    }
    .soft-header h1 {
        margin: 0;
        font-size: 2.2rem;
        font-weight: 900;
        background: linear-gradient(135deg, #1e3a8a, #7c3aed);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
    }
    .soft-header p {
        margin: 0.5rem 0 0 0;
        color: #64748b;
        font-size: 0.95rem;
        font-weight: 500;
    }
    
    /* Containers and cards */
    .container-box {
        background: rgba(255,255,255,0.88);
        border: 1px solid rgba(148,163,184,0.25);
        border-radius: 18px;
        padding: 1.5rem;
        margin-bottom: 1.5rem;
        box-shadow: 0 12px 28px rgba(15, 23, 42, 0.05);
        transition: all 0.3s ease;
    }
    .container-box:hover {
        box-shadow: 0 16px 40px rgba(15, 23, 42, 0.08);
        border-color: rgba(37, 99, 235, 0.2);
    }
    
    /* Metrics styling */
    div[data-testid="stMetricContainer"] > div {
        background: linear-gradient(135deg, rgba(255,255,255,0.95), rgba(239,246,255,0.8));
        border: 1px solid rgba(148,163,184,0.2);
        border-radius: 14px;
        padding: 1.5rem;
        box-shadow: 0 8px 20px rgba(15, 23, 42, 0.04);
        transition: all 0.3s ease;
    }
    div[data-testid="stMetricContainer"] > div:hover {
        transform: translateY(-2px);
        box-shadow: 0 12px 28px rgba(15, 23, 42, 0.06);
    }
    
    /* Sidebar */
    .stSidebar {
        background: linear-gradient(180deg, rgba(248,250,252,0.96), rgba(239,246,255,0.92));
    }
    .sidebar-divider {
        border-top: 1px solid rgba(148,163,184,0.15);
        margin: 1.5rem 0;
    }
    
    /* Buttons */
    .stButton > button {
        border-radius: 10px;
        background: linear-gradient(135deg, #2563eb, #1d4ed8);
        color: white;
        border: none;
        font-weight: 700;
        font-size: 0.95rem;
        padding: 0.6rem 1.2rem;
        box-shadow: 0 8px 20px rgba(37, 99, 235, 0.2);
        transition: all 0.3s ease;
    }
    .stButton > button:hover {
        background: linear-gradient(135deg, #1d4ed8, #1e40af);
        box-shadow: 0 12px 28px rgba(37, 99, 235, 0.3);
        transform: translateY(-1px);
    }
    .stDownloadButton > button {
        border-radius: 10px;
        border: 1.5px solid rgba(37, 99, 235, 0.3);
        background: white;
        color: #2563eb;
        font-weight: 600;
        box-shadow: 0 4px 12px rgba(37, 99, 235, 0.1);
        transition: all 0.3s ease;
    }
    .stDownloadButton > button:hover {
        background: rgba(37, 99, 235, 0.05);
        box-shadow: 0 6px 16px rgba(37, 99, 235, 0.15);
        transform: translateY(-1px);
    }
    
    /* File uploader styling */
    .stFileUploader {
        border-radius: 14px;
        overflow: hidden;
    }
    div[data-testid="stFileUploadDropzone"] {
        border-radius: 14px;
        border: 2px dashed rgba(37, 99, 235, 0.25);
        padding: 2rem;
        background: rgba(37, 99, 235, 0.02);
        transition: all 0.3s ease;
    }
    div[data-testid="stFileUploadDropzone"]:hover {
        border-color: rgba(37, 99, 235, 0.4);
        background: rgba(37, 99, 235, 0.05);
    }
    
    /* Text and typography */
    h1 { margin-top: 1.5rem; margin-bottom: 0.8rem; font-weight: 800; }
    h2 { margin-top: 1.2rem; margin-bottom: 0.7rem; font-weight: 800; }
    h3 { margin-top: 0.8rem; margin-bottom: 0.5rem; font-weight: 700; }
    
    /* Step indicators */
    .step-box {
        background: linear-gradient(135deg, rgba(37,99,235,0.08), rgba(124,58,237,0.08));
        border-left: 4px solid #2563eb;
        border-radius: 10px;
        padding: 1rem;
        margin-bottom: 1rem;
    }
    
    /* Status badges */
    .status-badge {
        display: inline-block;
        padding: 0.35rem 0.75rem;
        border-radius: 20px;
        font-size: 0.8rem;
        font-weight: 600;
        margin-right: 0.5rem;
    }
    .badge-compatible {
        background: #d1fae5;
        color: #065f46;
        border: 1px solid #a7f3d0;
    }
    .badge-conflict {
        background: #fee2e2;
        color: #7f1d1d;
        border: 1px solid #fecaca;
    }
    
    /* Audio player styling */
    .audio-section {
        background: linear-gradient(135deg, rgba(37,99,235,0.05), rgba(124,58,237,0.05));
        border-radius: 14px;
        padding: 1.5rem;
        border: 1px solid rgba(37,99,235,0.15);
    }
    
    /* Table styling */
    div[data-testid="stDataFrame"] {
        border-radius: 12px;
        overflow: hidden;
        box-shadow: 0 4px 12px rgba(15, 23, 42, 0.05);
    }
    
    /* Info, success, warning boxes */
    .stAlert {
        border-radius: 12px;
        border-left: 4px solid;
    }
    
    .element-container {
        margin-bottom: 1rem;
    }
    
    /* Radio buttons styling */
    .stRadio > label {
        padding: 0.5rem 0.75rem;
        border-radius: 8px;
        margin-bottom: 0.5rem;
        transition: all 0.2s;
    }
    
    /* Caption styling */
    .stCaption {
        color: #64748b;
        font-size: 0.85rem;
        line-height: 1.5;
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


def analyze_uploaded_image(image_path):
    """Create a lightweight semantic analysis for the uploaded image."""
    img = Image.open(image_path).convert('RGB')
    arr = np.array(img).astype(np.float32) / 255.0
    brightness = float(arr.mean())
    contrast = float(arr.std())
    r, g, b = arr[:, :, 0].mean(), arr[:, :, 1].mean(), arr[:, :, 2].mean()
    dominant_color = 'red' if max(r, g, b) == r else 'green' if max(r, g, b) == g else 'blue'

    if brightness > 0.68 and contrast < 0.18:
        scene = 'bright outdoor scene'
        objects = ['sky', 'open space', 'natural background']
        description = 'A bright outdoor scene with open space and natural lighting.'
    elif brightness < 0.35:
        scene = 'dim indoor or night scene'
        objects = ['dark background', 'indoor object', 'low-light setting']
        description = 'A low-light scene with a dark background and subdued visual context.'
    elif dominant_color == 'blue':
        scene = 'waterfront or natural environment'
        objects = ['water', 'shoreline', 'nature']
        description = 'A natural environment with blue tones suggesting water, sky, or a coastal scene.'
    elif dominant_color == 'red':
        scene = 'urban or performance setting'
        objects = ['stage light', 'crowd', 'city motion']
        description = 'A vivid urban or performance-like scene with warm color emphasis.'
    else:
        scene = 'general visual scene'
        objects = ['foreground object', 'background context', 'visual details']
        description = 'A general scene with balanced colors and visible visual structure.'

    return {
        'brightness': brightness,
        'contrast': contrast,
        'dominant_color': dominant_color,
        'scene': scene,
        'objects': objects,
        'description': description,
    }


def get_candidate_audio_concepts(scene, dominant_color):
    """Return candidate audio concepts with preliminary compatibility scores."""
    scene_lower = scene.lower()
    if 'outdoor' in scene_lower or 'water' in scene_lower or 'natural' in scene_lower or dominant_color == 'blue':
        return [
            {'concept': 'Ocean waves', 'score': 0.96, 'status': 'Compatible'},
            {'concept': 'Bird chirps', 'score': 0.88, 'status': 'Compatible'},
            {'concept': 'Piano music', 'score': 0.22, 'status': 'Conflict'},
            {'concept': 'Car engine', 'score': 0.18, 'status': 'Conflict'},
        ]
    if 'indoor' in scene_lower or 'performance' in scene_lower or 'night' in scene_lower:
        return [
            {'concept': 'Piano music', 'score': 0.94, 'status': 'Compatible'},
            {'concept': 'Audience applause', 'score': 0.90, 'status': 'Compatible'},
            {'concept': 'Car engine', 'score': 0.16, 'status': 'Conflict'},
            {'concept': 'Ocean waves', 'score': 0.12, 'status': 'Conflict'},
        ]
    if 'urban' in scene_lower or 'city' in scene_lower or dominant_color == 'red':
        return [
            {'concept': 'Car engine', 'score': 0.92, 'status': 'Compatible'},
            {'concept': 'Traffic ambience', 'score': 0.84, 'status': 'Compatible'},
            {'concept': 'Piano music', 'score': 0.15, 'status': 'Conflict'},
            {'concept': 'Ocean waves', 'score': 0.10, 'status': 'Conflict'},
        ]
    return [
        {'concept': 'Ambient synth', 'score': 0.74, 'status': 'Compatible'},
        {'concept': 'Soft piano', 'score': 0.68, 'status': 'Compatible'},
        {'concept': 'Car engine', 'score': 0.29, 'status': 'Conflict'},
        {'concept': 'Ocean waves', 'score': 0.21, 'status': 'Conflict'},
    ]


def filtered_verified_concepts(audio_candidates):
    compatible = [item for item in audio_candidates if item['status'] == 'Compatible']
    rejected = [item for item in audio_candidates if item['status'] == 'Conflict']
    return compatible, rejected


def semantic_verifier(scene, dominant_color, audio_candidates):
    """Resolve semantic mismatch by rejecting audio concepts that contradict the image scene."""
    scene_lower = scene.lower()
    compatible = []
    rejected = []
    reasons = []

    for item in audio_candidates:
        concept = item['concept'].lower()
        match = False
        reasoning = ""

        if ('water' in scene_lower or 'outdoor' in scene_lower or 'natural' in scene_lower or dominant_color == 'blue') and (
            'ocean' in concept or 'bird' in concept):
            match = True
            reasoning = f"✓ Scene is '{scene}' (water/outdoor/natural) → '{item['concept']}' is a NATURAL SOUND → MATCH!"
        elif ('indoor' in scene_lower or 'performance' in scene_lower or 'night' in scene_lower) and (
            'piano' in concept or 'audience' in concept):
            match = True
            reasoning = f"✓ Scene is '{scene}' (indoor/performance) → '{item['concept']}' is an INDOOR SOUND → MATCH!"
        elif ('urban' in scene_lower or 'city' in scene_lower or dominant_color == 'red') and (
            'car' in concept or 'traffic' in concept):
            match = True
            reasoning = f"✓ Scene is '{scene}' (urban/city) → '{item['concept']}' is an URBAN SOUND → MATCH!"
        elif 'ambient' in concept or 'soft' in concept:
            match = True
            reasoning = f"✓ '{item['concept']}' is UNIVERSAL/AMBIENT → Works with any scene → MATCH!"

        if item['status'] == 'Compatible' and match:
            compatible.append(item)
        elif item['status'] == 'Conflict' or not match:
            rejected.append(item)
            if item['status'] == 'Conflict':
                if 'ocean' in concept or 'bird' in concept:
                    if not ('water' in scene_lower or 'outdoor' in scene_lower or 'natural' in scene_lower or dominant_color == 'blue'):
                        reasoning = f"✗ Scene is '{scene}' (NOT water/outdoor/nature) → '{item['concept']}' is a NATURAL SOUND → MISMATCH!"
                elif 'piano' in concept or 'audience' in concept:
                    if not ('indoor' in scene_lower or 'performance' in scene_lower or 'night' in scene_lower):
                        reasoning = f"✗ Scene is '{scene}' (NOT indoor/performance) → '{item['concept']}' is an INDOOR SOUND → MISMATCH!"
                elif 'car' in concept or 'traffic' in concept:
                    if not ('urban' in scene_lower or 'city' in scene_lower or dominant_color == 'red'):
                        reasoning = f"✗ Scene is '{scene}' (NOT urban/city) → '{item['concept']}' is an URBAN SOUND → MISMATCH!"
                else:
                    reasoning = f"✗ '{item['concept']}' conflicts with the detected scene '{scene}' → REJECTED!"
            else:
                reasoning = f"✗ Scene is '{scene}' → '{item['concept']}' is not semantically consistent → REJECTED!"
            reasons.append(reasoning)

    return compatible, rejected, reasons

    return compatible, rejected, reasons


def concept_to_frequency(concept_name):
    concept_map = {
        'Piano music': 220,
        'Audience applause': 330,
        'Car engine': 110,
        'Traffic ambience': 140,
        'Ocean waves': 80,
        'Bird chirps': 440,
        'Ambient synth': 260,
        'Soft piano': 196,
    }
    return concept_map.get(concept_name, 220)


def generate_verified_audio(concepts, output_path='verified_audio.wav', sample_rate=22050, duration=3.0):
    """Generate a lightweight WAV from the verified concepts using simple sinusoidal synthesis."""
    t = np.linspace(0, duration, int(sample_rate * duration), endpoint=False)
    waveform = np.zeros_like(t)

    for concept in concepts:
        freq = concept_to_frequency(concept)
        amp = 0.25 + concept['score'] * 0.35
        waveform += amp * np.sin(2 * np.pi * freq * t)

    waveform = waveform / (np.max(np.abs(waveform)) + 1e-8)
    audio_int16 = (waveform * 32767).astype(np.int16)
    wav.write(output_path, sample_rate, audio_int16)
    return output_path

# ============================================================================
# MAIN UI
# ============================================================================

# Header
st.markdown('''
<div class="soft-header">
    <div class="brand">CAS-V</div>
    <h1>Semantic audio generation with verification</h1>
    <p>Conflict-aware semantic verification for image-to-audio generation</p>
</div>
''', unsafe_allow_html=True)

# Sidebar Navigation
st.sidebar.markdown("## Navigation")
page = st.sidebar.radio(
    "Select page:",
    ["Home", "Audio Generation", "Experimentation Results", "Settings"],
    label_visibility="collapsed"
)

st.sidebar.caption("A clean workflow for image upload, compatibility checks, and verified sound output.")

# ============================================================================
# PAGE: HOME
# ============================================================================
if page == "Home":
    st.markdown("### Image-to-Audio CAS-V Workflow")
    st.caption("📤 Upload an image to analyze the scene, verify compatible audio concepts, and generate verified audio output.")

    uploaded_file = st.file_uploader("Choose an image (JPG, JPEG, PNG)", type=['jpg', 'jpeg', 'png'], key='workflow_upload')

    if uploaded_file is not None:
        image_bytes = uploaded_file.getvalue()
        temp_path = os.path.join('uploaded_images', uploaded_file.name)
        os.makedirs('uploaded_images', exist_ok=True)
        with open(temp_path, 'wb') as f:
            f.write(image_bytes)

        # Step 1: Image Analysis
        with st.container(border=False):
            st.markdown("---")
            st.markdown("#### Step 1️⃣ Image Analysis")
            
            col_img, col_meta = st.columns([1, 1.2], gap="large")
            with col_img:
                st.markdown('<div class="container-box">', unsafe_allow_html=True)
                image = Image.open(temp_path).convert('RGB')
                st.image(image, width="stretch", caption='✓ Uploaded image')
                st.markdown('</div>', unsafe_allow_html=True)

            with col_meta:
                st.markdown('<div class="container-box">', unsafe_allow_html=True)
                analysis = analyze_uploaded_image(temp_path)
                st.markdown("**📋 Scene Summary**")
                st.write(f"🎬 **Scene:** {analysis['scene']}")
                st.write(f"🎨 **Dominant color:** {analysis['dominant_color']}")
                st.write(f"🔍 **Objects:** {', '.join(analysis['objects'])}")
                st.write(f"☀️ **Brightness:** {analysis['brightness']:.3f}")
                st.write(f"📊 **Contrast:** {analysis['contrast']:.3f}")
                st.divider()
                st.caption(f"💭 {analysis['description']}")
                st.markdown('</div>', unsafe_allow_html=True)

        # Step 2: Candidate Concepts
        st.markdown("---")
        st.markdown("#### Step 2️⃣ Candidate Audio Concepts")
        candidates = get_candidate_audio_concepts(analysis['scene'], analysis['dominant_color'])
        compatible, rejected = filtered_verified_concepts(candidates)
        
        df_candidates = pd.DataFrame(candidates)
        st.dataframe(df_candidates, hide_index=True, width="stretch", use_container_width=True)

        # Step 3: Semantic Verification
        st.markdown("---")
        st.markdown("#### Step 3️⃣ Semantic Verification — Detecting & Resolving Conflicts")
        
        verified_compatible, filtered_conflicts, verifier_reasons = semantic_verifier(
            analysis['scene'], analysis['dominant_color'], candidates
        )
        
        st.markdown('<div class="container-box" style="border-left: 4px solid #3b82f6;">', unsafe_allow_html=True)
        st.markdown("**🔍 How the Verifier Works**")
        st.caption("""
        The semantic verifier checks each candidate audio concept against the scene to make sure they match. 
        It asks: **"Does this sound belong in this image?"**
        
        For example:
        - 🌊 Ocean waves → only accept for water/beach/nature scenes
        - 🎹 Piano music → only accept for indoor/concert/performance scenes  
        - 🚗 Car engine → only accept for urban/city scenes
        """)
        st.markdown('</div>', unsafe_allow_html=True)
        
        # Show the verification logic for each concept
        st.markdown("**📋 Detailed Verification Logic:**")
        
        col_left, col_right = st.columns(2, gap="medium")
        
        # Show compatible concepts with reasoning
        with col_left:
            st.markdown('<div class="container-box" style="border-left: 4px solid #10b981;">', unsafe_allow_html=True)
            st.markdown("**✓ ACCEPTED Concepts**")
            if verified_compatible:
                for item in verified_compatible:
                    # Find the reasoning for this item
                    reasoning = next((r for r in verifier_reasons if item['concept'].lower() in r.lower() and '✓' in r), None)
                    
                    st.markdown(f"""
                    <div style="background: rgba(16, 185, 129, 0.08); border-left: 3px solid #10b981; padding: 0.8rem; border-radius: 6px; margin-bottom: 0.8rem;">
                    <strong style="color: #10b981;">✓ {item['concept']}</strong><br>
                    <span style="font-size: 0.85rem; color: #047857;">{reasoning if reasoning else f"Compatible with {analysis['scene']}"}</span>
                    </div>
                    """, unsafe_allow_html=True)
            else:
                st.write("🚫 No compatible concepts found.")
            st.markdown('</div>', unsafe_allow_html=True)
        
        # Show rejected concepts with reasoning
        with col_right:
            st.markdown('<div class="container-box" style="border-left: 4px solid #ef4444;">', unsafe_allow_html=True)
            st.markdown("**✗ REJECTED Concepts**")
            if filtered_conflicts:
                for item in filtered_conflicts:
                    # Find the reasoning for this item
                    reasoning = next((r for r in verifier_reasons if item['concept'].lower() in r.lower() and '✗' in r), None)
                    
                    st.markdown(f"""
                    <div style="background: rgba(239, 68, 68, 0.08); border-left: 3px solid #ef4444; padding: 0.8rem; border-radius: 6px; margin-bottom: 0.8rem;">
                    <strong style="color: #ef4444;">✗ {item['concept']}</strong><br>
                    <span style="font-size: 0.85rem; color: #991b1b;">{reasoning if reasoning else f"Doesn't match {analysis['scene']}"}</span>
                    </div>
                    """, unsafe_allow_html=True)
            else:
                st.write("✓ No conflicts detected.")
            st.markdown('</div>', unsafe_allow_html=True)

        # Step 4: Verified Results
        st.markdown("---")
        st.markdown("#### Step 4️⃣ Verification Results")
        
        col_compatible, col_rejected = st.columns(2, gap="medium")
        with col_compatible:
            st.markdown('<div class="container-box" style="border-left: 4px solid #10b981;">', unsafe_allow_html=True)
            if verified_compatible:
                st.success(f"✓ {len(verified_compatible)} Compatible Concept(s) Retained")
                for item in verified_compatible:
                    score_pct = int(item['score'] * 100)
                    st.write(f"• **{item['concept']}** — {score_pct}% match")
            else:
                st.warning("⚠️ No compatible concepts were found for this image.")
            st.markdown('</div>', unsafe_allow_html=True)
            
        with col_rejected:
            st.markdown('<div class="container-box" style="border-left: 4px solid #ef4444;">', unsafe_allow_html=True)
            if filtered_conflicts:
                st.warning(f"✗ {len(filtered_conflicts)} Conflict(s) Filtered Out")
                for item in filtered_conflicts:
                    score_pct = int(item['score'] * 100)
                    st.write(f"• **{item['concept']}** — {score_pct}% (rejected)")
            else:
                st.info("✓ No conflicts detected.")
            st.markdown('</div>', unsafe_allow_html=True)

        # Step 5: Audio Generation
        st.markdown("---")
        st.markdown("#### Step 5️⃣ Verified Audio Output")
        
        if verified_compatible:
            st.markdown('<div class="audio-section">', unsafe_allow_html=True)
            st.markdown("**🎵 Listen to the verified audio:**")
            verified_path = generate_verified_audio(verified_compatible, output_path='verified_audio.wav')
            st.audio(verified_path, format='audio/wav')
            
            col_audio_left, col_audio_right = st.columns([3, 1])
            with col_audio_right:
                with open(verified_path, 'rb') as audio_file:
                    st.download_button(
                        '📥 Download',
                        audio_file.read(),
                        file_name='verified_audio.wav',
                        mime='audio/wav',
                        use_container_width=True
                    )
            st.markdown('</div>', unsafe_allow_html=True)
        else:
            st.markdown('<div class="container-box" style="border-left: 4px solid #f59e0b;">', unsafe_allow_html=True)
            st.caption("⚠️ Verified audio generation is unavailable because no compatible concept passed the semantic safety check.")
            st.markdown('</div>', unsafe_allow_html=True)

    else:
        st.markdown("---")
        st.markdown("##  What This Research Is Solving")
        
        st.markdown('<div class="container-box">', unsafe_allow_html=True)
        st.markdown("""
        **The Problem:** Current image-to-audio systems can generate realistic sounds, but they often produce audio that doesn't match the actual content or context of the input image.
        
        **Example:** An image of a person playing piano might incorrectly generate car engine or ocean wave sounds.
        
        **Root Cause:** Many systems generate audio without a semantic verification step to ensure consistency with the visual scene.
        
        **Our Solution (CAS-V):** We address this by analyzing the image, extracting its semantic description, comparing it with candidate audio concepts, detecting conflicts, and filtering incompatible concepts before generating the final audio output.
        """)
        st.markdown('</div>', unsafe_allow_html=True)

        st.markdown("---")
        st.markdown("## How the CAS-V System Works")
        
        steps = st.columns(5, gap="small")
        
        step_data = [
            ("1️⃣", "Input", "Upload an image\nfrom the browser"),
            ("2️⃣", "Analyze", "Extract brightness,\ncontrast, and scene\nsemantics"),
            ("3️⃣", "Generate", "Create candidate\naudio concepts"),
            ("4️⃣", "Verify", "Check compatibility\nand reject conflicts"),
            ("5️⃣", "Output", "Generate verified\naudio only"),
        ]
        
        for idx, (col, (emoji, title, desc)) in enumerate(zip(steps, step_data)):
            with col:
                st.markdown(f'''
                <div class="container-box" style="text-align: center; border-left: none;">
                <div style="font-size: 2rem; margin-bottom: 0.5rem;">{emoji}</div>
                <div style="font-weight: 700; font-size: 0.95rem; margin-bottom: 0.5rem;">{title}</div>
                <div style="font-size: 0.8rem; color: #64748b; line-height: 1.4;">{desc}</div>
                </div>
                ''', unsafe_allow_html=True)

        st.markdown("---")
        st.markdown("##  Workflow Overview")
        
        st.markdown('''
        <div class="step-box">
        <strong>🔄 The CAS-V Pipeline:</strong><br>
        <br>
        <strong>1. Input:</strong> User uploads an image via the interface.<br>
        <strong>2. Scene Analysis:</strong> Convert visual content into a semantic scene summary (brightness, contrast, color, objects).<br>
        <strong>3. Candidate Generation:</strong> Generate candidate audio concepts (e.g., waves, engine, piano, birds).<br>
        <strong>4. Conflict Detection:</strong> Compare candidates against image context and detect semantic mismatches.<br>
        <strong>5. Verified Synthesis:</strong> Use only compatible concepts to produce the final audio waveform.
        </div>
        ''', unsafe_allow_html=True)

        st.markdown("---")
        st.markdown("##  Technology Stack & Future Enhancements")
        
        col_current, col_future = st.columns(2, gap="medium")
        
        with col_current:
            st.markdown('<div class="container-box" style="border-left: 4px solid #3b82f6;">', unsafe_allow_html=True)
            st.markdown("**🏗️ Current Implementation**")
            st.markdown("""
            The current CAS-V system uses:
            
            - **Computer Vision** – Image analysis (brightness, contrast, color)
            - **Rule-Based Matching** – Keyword-based semantic verification
            - **Signal Processing** – Audio waveform synthesis
            
            **Advantage:** Fast, interpretable, no deep learning needed
            
            **Limitation:** Simple keyword matching, not true semantic understanding
            """)
            st.markdown('</div>', unsafe_allow_html=True)
        
        with col_future:
            st.markdown('<div class="container-box" style="border-left: 4px solid #8b5cf6;">', unsafe_allow_html=True)
            st.markdown("**🚀 True NLP Enhancement (Future)**")
            st.markdown("""
            To implement true NLP-based semantic verification:
            
            1. **Semantic Embeddings** – Convert text to vectors (Word2Vec, FastText)
            2. **CLIP Models** – Multimodal understanding of images + concepts
            3. **Fine-tuned Verifier** – Train on labeled image-audio pairs
            4. **Transformer Networks** – BERT, GPT for real language understanding
            
            **Advantage:** Deeper semantic reasoning, handles novel concepts
            
            **Trade-off:** Requires training data, more compute
            """)
            st.markdown('</div>', unsafe_allow_html=True)

        st.markdown("---")
        st.markdown("##  Research Metrics")
        
        results_file = "paper_results/generation_results.json"
        if os.path.exists(results_file):
            results = load_json_results(results_file)
            if results:
                metrics = results.get('quality_metrics', {})
                col_m1, col_m2, col_m3 = st.columns(3, gap="medium")
                
                with col_m1:
                    fad = metrics.get('frechet_audio_distance', {}).get('value', 'N/A')
                    st.metric("🎵 FAD Score", fad, help="Frechet Audio Distance (lower is better)")
                
                with col_m2:
                    snr = metrics.get('signal_to_noise_ratio', {}).get('value', 'N/A')
                    st.metric("📢 SNR (dB)", snr, help="Signal-to-Noise Ratio (higher is better)")
                
                with col_m3:
                    clap = metrics.get('clap_alignment_score', {}).get('value', 'N/A')
                    st.metric("🔗 CLAP Score", clap, help="CLAP Alignment Score (higher is better)")
        else:
            st.info("📊 Metrics will appear here after generating results.")

# ============================================================================
# PAGE: AUDIO GENERATION
# ============================================================================
elif page == "Audio Generation":
    st.markdown("## 🎵 CAS-V Audio Generation")
    st.caption("Upload an image to analyze its content, verify semantic compatibility, and generate semantically relevant audio.")

    uploaded_file = st.file_uploader("Choose an image for CAS-V processing", type=['jpg', 'jpeg', 'png'], key='audio_upload')
    if uploaded_file is not None:
        image_path = os.path.join('uploaded_images', uploaded_file.name)
        os.makedirs('uploaded_images', exist_ok=True)
        with open(image_path, 'wb') as f:
            f.write(uploaded_file.getbuffer())

        analysis = analyze_uploaded_image(image_path)
        candidates = get_candidate_audio_concepts(analysis['scene'], analysis['dominant_color'])
        compatible, rejected = filtered_verified_concepts(candidates)
        verified_compatible, filtered_conflicts, verifier_reasons = semantic_verifier(
            analysis['scene'], analysis['dominant_color'], candidates
        )

        col1, col2 = st.columns([1, 1.2], gap="large")
        with col1:
            st.markdown('<div class="container-box">', unsafe_allow_html=True)
            st.image(Image.open(image_path), width="stretch", caption='✓ Uploaded Image')
            st.markdown('</div>', unsafe_allow_html=True)
            
        with col2:
            st.markdown('<div class="container-box">', unsafe_allow_html=True)
            st.markdown("### Image Analysis")
            st.markdown(f"**🎬 Scene:** {analysis['scene']}")
            st.markdown(f"**🎨 Color:** {analysis['dominant_color']}")
            st.markdown(f"**📋 Objects:** {', '.join(analysis['objects'])}")
            st.divider()
            st.markdown(f"**📊 Brightness:** {analysis['brightness']:.3f}")
            st.markdown(f"**📊 Contrast:** {analysis['contrast']:.3f}")
            st.divider()
            st.caption(analysis['description'])
            st.markdown('</div>', unsafe_allow_html=True)

        st.markdown("---")
        st.markdown("### Candidate Audio Concepts")
        st.dataframe(pd.DataFrame(candidates), hide_index=True, width="stretch", use_container_width=True)

        st.markdown("---")
        st.markdown("### Semantic Verification Results")
        st.markdown("**🔍 How the verifier works:** It checks if each audio concept matches the detected scene.")
        
        col_a, col_b = st.columns(2, gap="medium")
        with col_a:
            st.markdown('<div class="container-box" style="border-left: 4px solid #10b981;">', unsafe_allow_html=True)
            st.markdown("**✓ ACCEPTED Concepts**")
            if verified_compatible:
                for item in verified_compatible:
                    reasoning = next((r for r in verifier_reasons if item['concept'].lower() in r.lower() and '✓' in r), None)
                    st.markdown(f"""
                    <div style="background: rgba(16, 185, 129, 0.08); border-left: 3px solid #10b981; padding: 0.7rem; border-radius: 6px; margin-bottom: 0.6rem;">
                    <strong style="color: #10b981;">✓ {item['concept']}</strong><br>
                    <span style="font-size: 0.8rem; color: #047857;">{reasoning if reasoning else 'Compatible with scene'}</span>
                    </div>
                    """, unsafe_allow_html=True)
            else:
                st.write("No compatible concepts found.")
            st.markdown('</div>', unsafe_allow_html=True)
            
        with col_b:
            st.markdown('<div class="container-box" style="border-left: 4px solid #ef4444;">', unsafe_allow_html=True)
            st.markdown("**✗ REJECTED Concepts**")
            if filtered_conflicts:
                for item in filtered_conflicts:
                    reasoning = next((r for r in verifier_reasons if item['concept'].lower() in r.lower() and '✗' in r), None)
                    st.markdown(f"""
                    <div style="background: rgba(239, 68, 68, 0.08); border-left: 3px solid #ef4444; padding: 0.7rem; border-radius: 6px; margin-bottom: 0.6rem;">
                    <strong style="color: #ef4444;">✗ {item['concept']}</strong><br>
                    <span style="font-size: 0.8rem; color: #991b1b;">{reasoning if reasoning else 'Conflicts with scene'}</span>
                    </div>
                    """, unsafe_allow_html=True)
            else:
                st.write("No conflicts detected.")
            st.markdown('</div>', unsafe_allow_html=True)

        if verified_compatible:
            st.markdown("---")
            st.markdown("### Audio Generation")
            st.markdown('<div class="audio-section">', unsafe_allow_html=True)
            verified_path = generate_verified_audio(verified_compatible, output_path='verified_audio.wav')
            st.audio(verified_path, format='audio/wav')
            
            col_dl, col_space = st.columns([1, 3])
            with col_dl:
                with open(verified_path, 'rb') as f:
                    st.download_button(
                        '📥 Download Audio',
                        f.read(),
                        file_name='verified_audio.wav',
                        mime='audio/wav',
                        use_container_width=True
                    )
            st.markdown('</div>', unsafe_allow_html=True)
    else:
        st.markdown('<div class="container-box">', unsafe_allow_html=True)
        st.info('📤 Upload an image to begin the CAS-V workflow.')
        st.markdown('</div>', unsafe_allow_html=True)

# ============================================================================
# PAGE: EXPERIMENTATION RESULTS
# ============================================================================
elif page == "Experimentation Results":
    st.markdown("---")
    st.markdown("## 🧪 Experimentation Results")
    st.caption("View results from actual system runs, audio samples, and performance metrics.")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown("### 📂 Generated Audio & Analysis")
        
        # Check for uploaded images and generated audio
        uploaded_dir = "uploaded_images"
        if os.path.exists(uploaded_dir):
            uploaded_files = sorted(glob.glob(f"{uploaded_dir}/*"))
            
            if uploaded_files:
                st.markdown("**Sample Images & Verified Audio Output:**")
                
                for idx, img_path in enumerate(uploaded_files):
                    if os.path.isfile(img_path) and img_path.lower().endswith(('.png', '.jpg', '.jpeg')):
                        try:
                            st.markdown(f"#### Sample {idx + 1}: {os.path.basename(img_path)}")
                            
                            col_img, col_audio = st.columns([1, 1], gap="medium")
                            
                            with col_img:
                                img = Image.open(img_path)
                                st.image(img, use_column_width=True, caption="Input Image")
                            
                            with col_audio:
                                st.markdown("**Analysis Result:**")
                                analysis = analyze_uploaded_image(img_path)
                                st.write(f"🎬 **Scene:** {analysis['scene']}")
                                st.write(f"🎨 **Color:** {analysis['dominant_color']}")
                                st.write(f"☀️ **Brightness:** {analysis['brightness']:.3f}")
                                st.write(f"📊 **Contrast:** {analysis['contrast']:.3f}")
                                
                                # Check if verified audio exists
                                verified_audio = "verified_audio.wav"
                                if os.path.exists(verified_audio):
                                    st.markdown("**Generated Audio:**")
                                    st.audio(verified_audio, format='audio/wav')
                            
                            st.divider()
                        except Exception as e:
                            st.warning(f"Could not process {img_path}: {str(e)}")
            else:
                st.info("📤 No uploaded images yet. Start by uploading an image on the Audio Generation page.")
        else:
            st.info("📤 No experimentation data yet. Upload an image to begin.")
    
    with col2:
        st.markdown("### 📊 Summary Statistics")
        
        # Count results
        uploaded_dir = "uploaded_images"
        if os.path.exists(uploaded_dir):
            images = [f for f in glob.glob(f"{uploaded_dir}/*") if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
            st.metric("📸 Images Processed", len(images))
        else:
            st.metric("📸 Images Processed", 0)
        
        verified_audio = "verified_audio.wav"
        if os.path.exists(verified_audio):
            st.metric("🎵 Audio Generated", "✓ Yes")
        else:
            st.metric("🎵 Audio Generated", "⏳ Pending")
        
        st.markdown("### 🎯 Actions")
        
        if st.button("🔄 Refresh Results", use_container_width=True):
            st.rerun()
        
        st.markdown("### 📥 Export Options")
        
        # Export verified audio
        if os.path.exists("verified_audio.wav"):
            with open("verified_audio.wav", "rb") as f:
                st.download_button(
                    label="📥 Download Latest Audio",
                    data=f.read(),
                    file_name="verified_audio.wav",
                    mime="audio/wav",
                    use_container_width=True
                )
        
        # Export results JSON if available
        results_file = "paper_results/generation_results.json"
        if os.path.exists(results_file):
            with open(results_file, "rb") as f:
                st.download_button(
                    label="📊 Download Metadata JSON",
                    data=f.read(),
                    file_name="paper_metadata.json",
                    mime="application/json"
                )

# ============================================================================
# PAGE: SETTINGS
# ============================================================================
elif page == "Settings":
    st.markdown("## Settings & Configuration")
    
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
