import streamlit as st
import os
import glob
import json
import uuid
import sys
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd
from PIL import Image
from scipy.io.wavfile import write as wav_write


# ============================================================================
# PAGE CONFIGURATION
# ============================================================================

st.set_page_config(
    page_title="CAS-V Research Prototype",
    page_icon="🎵",
    layout="wide",
    initial_sidebar_state="expanded"
)


# ============================================================================
# PROJECT DIRECTORIES
# ============================================================================

BASE_DIR = Path.cwd()

UPLOAD_DIR = BASE_DIR / "uploaded_images"
EXPERIMENT_DIR = BASE_DIR / "experiments"

UPLOAD_DIR.mkdir(exist_ok=True)
EXPERIMENT_DIR.mkdir(exist_ok=True)


# ============================================================================
# SESSION STATE
# ============================================================================

DEFAULTS = {
    "sample_rate": 22050,
    "audio_duration": 3.0,
    "last_result": None,
    "last_image_path": None,
    "last_analysis": None,
    "last_candidates": None,
    "last_verification": None,
    "last_audio_path": None,
}

for key, value in DEFAULTS.items():
    if key not in st.session_state:
        st.session_state[key] = value


# ============================================================================
# CUSTOM CSS
# ============================================================================

st.markdown(
    """
<style>

/* -------------------------------------------------------------------------
   GLOBAL
------------------------------------------------------------------------- */

.stApp {
    background:
        radial-gradient(circle at top left,
            rgba(37, 99, 235, 0.10),
            transparent 28%),
        radial-gradient(circle at top right,
            rgba(124, 58, 237, 0.08),
            transparent 28%),
        #f8fafc;
}

.main .block-container {
    max-width: 1450px;
    padding-top: 2rem;
    padding-bottom: 3rem;
}


/* -------------------------------------------------------------------------
   HERO
------------------------------------------------------------------------- */

.casv-hero {
    background:
        linear-gradient(
            135deg,
            rgba(255,255,255,0.98),
            rgba(239,246,255,0.95)
        );

    border: 1px solid rgba(148,163,184,0.22);

    border-radius: 24px;

    padding: 2.2rem 2.5rem;

    margin-bottom: 2rem;

    box-shadow:
        0 20px 45px rgba(15,23,42,0.07);
}

.casv-badge {
    display: inline-block;

    background:
        linear-gradient(
            135deg,
            #2563eb,
            #7c3aed
        );

    color: white;

    border-radius: 999px;

    padding: 0.4rem 0.9rem;

    font-size: 0.72rem;

    font-weight: 800;

    letter-spacing: 0.12em;

    margin-bottom: 0.8rem;
}

.casv-title {
    font-size: 2.4rem;

    font-weight: 900;

    margin: 0;

    background:
        linear-gradient(
            135deg,
            #1e3a8a,
            #7c3aed
        );

    -webkit-background-clip: text;

    -webkit-text-fill-color: transparent;
}

.casv-subtitle {
    color: #64748b;

    margin-top: 0.7rem;

    font-size: 1rem;
}


/* -------------------------------------------------------------------------
   CARDS
------------------------------------------------------------------------- */

.research-card {
    background:
        rgba(255,255,255,0.92);

    border:
        1px solid rgba(148,163,184,0.20);

    border-radius: 18px;

    padding: 1.4rem;

    margin-bottom: 1rem;

    box-shadow:
        0 8px 24px rgba(15,23,42,0.04);
}

.research-card:hover {

    box-shadow:
        0 14px 30px rgba(15,23,42,0.07);
}

.accept-card {

    background:
        rgba(236,253,245,0.75);

    border-left:
        5px solid #10b981;

    border-radius:
        12px;

    padding:
        1rem;

    margin-bottom:
        0.8rem;
}

.reject-card {

    background:
        rgba(254,242,242,0.75);

    border-left:
        5px solid #ef4444;

    border-radius:
        12px;

    padding:
        1rem;

    margin-bottom:
        0.8rem;
}

.info-card {

    background:
        rgba(239,246,255,0.75);

    border-left:
        5px solid #3b82f6;

    border-radius:
        12px;

    padding:
        1rem;

    margin-bottom:
        1rem;
}


/* -------------------------------------------------------------------------
   PIPELINE
------------------------------------------------------------------------- */

.pipeline-card {

    text-align:
        center;

    background:
        rgba(255,255,255,0.92);

    border:
        1px solid rgba(148,163,184,0.18);

    border-radius:
        18px;

    padding:
        1.3rem;

    min-height:
        170px;
}

.pipeline-icon {

    font-size:
        2rem;

    margin-bottom:
        0.6rem;
}

.pipeline-title {

    font-weight:
        800;

    margin-bottom:
        0.5rem;
}

.pipeline-desc {

    color:
        #64748b;

    font-size:
        0.82rem;
}


/* -------------------------------------------------------------------------
   CAS-V RESULT
------------------------------------------------------------------------- */

.verification-summary {

    background:
        linear-gradient(
            135deg,
            rgba(37,99,235,0.08),
            rgba(124,58,237,0.08)
        );

    border:
        1px solid rgba(99,102,241,0.18);

    border-radius:
        18px;

    padding:
        1.5rem;

    margin:
        1rem 0;
}

.verification-title {

    font-weight:
        900;

    font-size:
        1.2rem;

    color:
        #1e3a8a;

    margin-bottom:
        0.5rem;
}


/* -------------------------------------------------------------------------
   METRICS
------------------------------------------------------------------------- */

div[data-testid="stMetricContainer"] {

    background:
        rgba(255,255,255,0.92);

    border:
        1px solid rgba(148,163,184,0.18);

    border-radius:
        15px;

    padding:
        1rem;

    box-shadow:
        0 6px 18px rgba(15,23,42,0.04);
}


/* -------------------------------------------------------------------------
   BUTTONS
------------------------------------------------------------------------- */

.stButton > button {

    border-radius:
        10px;

    background:
        linear-gradient(
            135deg,
            #2563eb,
            #1d4ed8
        );

    color:
        white;

    border:
        none;

    font-weight:
        700;

    padding:
        0.6rem 1rem;
}

.stButton > button:hover {

    transform:
        translateY(-1px);
}


/* -------------------------------------------------------------------------
   FILE UPLOAD
------------------------------------------------------------------------- */

div[data-testid="stFileUploadDropzone"] {

    border:
        2px dashed rgba(37,99,235,0.30);

    border-radius:
        16px;

    background:
        rgba(37,99,235,0.025);

    padding:
        2rem;
}


/* -------------------------------------------------------------------------
   SIDEBAR
------------------------------------------------------------------------- */

.stSidebar {

    background:
        linear-gradient(
            180deg,
            #f8fafc,
            #eef2ff
        );
}

</style>
""",
    unsafe_allow_html=True
)


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def safe_filename(name):
    """
    Create a safe filename.
    """

    return "".join(
        c if c.isalnum() or c in "._-" else "_"
        for c in name
    )


def save_uploaded_image(uploaded_file):
    """
    Save uploaded image with a unique filename.
    """

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    filename = (
        f"{timestamp}_"
        f"{uuid.uuid4().hex[:8]}_"
        f"{safe_filename(uploaded_file.name)}"
    )

    image_path = UPLOAD_DIR / filename

    with open(image_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    return image_path


# ============================================================================
# VISUAL ANALYSIS
# ============================================================================

def analyze_uploaded_image(image_path):
    """
    Lightweight visual-statistics analysis.

    IMPORTANT:
    This prototype does NOT perform real object detection.
    It calculates image statistics and produces a heuristic scene hypothesis.
    """

    image = Image.open(image_path).convert("RGB")

    array = (
        np.array(image)
        .astype(np.float32)
        / 255.0
    )

    brightness = float(array.mean())

    contrast = float(array.std())

    red_mean = float(array[:, :, 0].mean())
    green_mean = float(array[:, :, 1].mean())
    blue_mean = float(array[:, :, 2].mean())

    color_values = {
        "red": red_mean,
        "green": green_mean,
        "blue": blue_mean,
    }

    dominant_color = max(
        color_values,
        key=color_values.get
    )

    # ----------------------------------------------------------------------
    # HEURISTIC SCENE HYPOTHESIS
    # ----------------------------------------------------------------------

    if brightness > 0.68 and contrast < 0.18:

        scene = "bright outdoor / open scene"

        confidence = "Low"

        description = (
            "The image has high overall brightness and relatively low "
            "visual contrast. The prototype therefore hypothesizes an "
            "open or brightly illuminated environment."
        )

    elif brightness < 0.35:

        scene = "dim / low-light scene"

        confidence = "Low"

        description = (
            "The image has low overall brightness. The prototype "
            "therefore hypothesizes a dim, indoor, or night-like "
            "visual environment."
        )

    elif dominant_color == "blue":

        scene = "blue-dominant natural / open environment"

        confidence = "Low"

        description = (
            "Blue is the strongest average color component. "
            "The prototype uses this as a heuristic signal for a "
            "possible open, sky-like, water-like, or natural environment."
        )

    elif dominant_color == "red":

        scene = "warm-color dominant environment"

        confidence = "Low"

        description = (
            "Red is the strongest average color component. "
            "The prototype interprets this only as a warm-color "
            "visual environment; it does not detect specific objects."
        )

    else:

        scene = "general visual environment"

        confidence = "Low"

        description = (
            "The image does not strongly match one of the prototype's "
            "simple visual heuristic categories."
        )

    return {
        "brightness": brightness,
        "contrast": contrast,
        "dominant_color": dominant_color,
        "scene": scene,
        "confidence": confidence,
        "description": description,
        "method": (
            "Visual statistics + rule-based scene hypothesis"
        )
    }


# ============================================================================
# CANDIDATE AUDIO GENERATION
# ============================================================================

def get_candidate_audio_concepts():
    """
    Return a neutral candidate pool.

    IMPORTANT:
    Concepts are NOT pre-labeled Compatible or Conflict.
    CAS-V makes that decision later.
    """

    return [
        {
            "concept": "Ocean Waves",
            "score": 0.96
        },
        {
            "concept": "Bird Chirps",
            "score": 0.88
        },
        {
            "concept": "Piano Music",
            "score": 0.74
        },
        {
            "concept": "Audience Applause",
            "score": 0.68
        },
        {
            "concept": "Car Engine",
            "score": 0.61
        },
        {
            "concept": "Traffic Ambience",
            "score": 0.57
        },
        {
            "concept": "Ambient Synth",
            "score": 0.52
        },
        {
            "concept": "Soft Piano",
            "score": 0.48
        }
    ]


# ============================================================================
# CONCEPT CATEGORIES
# ============================================================================

CONCEPT_CATEGORIES = {

    "Ocean Waves":
        "natural",

    "Bird Chirps":
        "natural",

    "Piano Music":
        "performance",

    "Audience Applause":
        "performance",

    "Car Engine":
        "urban",

    "Traffic Ambience":
        "urban",

    "Ambient Synth":
        "ambient",

    "Soft Piano":
        "ambient"
}


# ============================================================================
# SCENE CATEGORY INFERENCE
# ============================================================================

def infer_scene_category(
    scene,
    dominant_color
):
    """
    Map heuristic scene labels to broad semantic categories.
    """

    scene_lower = scene.lower()

    if (
        "blue-dominant" in scene_lower
        or "natural" in scene_lower
    ):

        return "natural"

    if (
        "bright outdoor" in scene_lower
        or "open scene" in scene_lower
    ):

        return "natural"

    if (
        "dim" in scene_lower
        or "low-light" in scene_lower
    ):

        return "performance"

    if (
        "warm-color" in scene_lower
        or dominant_color == "red"
    ):

        return "urban"

    return "general"


# ============================================================================
# CAS-V SEMANTIC VERIFIER
# ============================================================================

def semantic_verifier(
    analysis,
    audio_candidates
):
    """
    CAS-V prototype verifier.

    This version returns STRUCTURED decisions.
    Every candidate receives:

    - Concept
    - Candidate score
    - Scene category
    - Concept category
    - Decision
    - Explanation
    """

    scene = analysis["scene"]

    dominant_color = analysis["dominant_color"]

    scene_category = infer_scene_category(
        scene,
        dominant_color
    )

    verified_results = []

    compatible = []

    rejected = []

    for item in audio_candidates:

        concept = item["concept"]

        candidate_score = item["score"]

        concept_category = CONCEPT_CATEGORIES.get(
            concept,
            "unknown"
        )

        decision = "Rejected"

        reason = ""

        # ------------------------------------------------------------------
        # SEMANTIC RULES
        # ------------------------------------------------------------------

        if concept_category == "ambient":

            decision = "Accepted"

            reason = (
                "Ambient concepts are treated as broadly compatible "
                "with the current prototype scene hypothesis."
            )

        elif (
            scene_category == "natural"
            and concept_category == "natural"
        ):

            decision = "Accepted"

            reason = (
                "The heuristic scene category is NATURAL/OPEN and "
                "the candidate belongs to the NATURAL sound category."
            )

        elif (
            scene_category == "performance"
            and concept_category == "performance"
        ):

            decision = "Accepted"

            reason = (
                "The heuristic scene category is DIM/PERFORMANCE and "
                "the candidate belongs to the PERFORMANCE sound category."
            )

        elif (
            scene_category == "urban"
            and concept_category == "urban"
        ):

            decision = "Accepted"

            reason = (
                "The heuristic scene category is WARM/URBAN and "
                "the candidate belongs to the URBAN sound category."
            )

        else:

            decision = "Rejected"

            reason = (
                f"The candidate belongs to the {concept_category.upper()} "
                f"sound category, while the current heuristic scene "
                f"category is {scene_category.upper()}."
            )

        result = {
            "concept": concept,
            "score": candidate_score,
            "scene_category": scene_category,
            "concept_category": concept_category,
            "decision": decision,
            "reason": reason
        }

        verified_results.append(result)

        if decision == "Accepted":
            compatible.append(result)

        else:
            rejected.append(result)

    return {
        "scene_category": scene_category,
        "results": verified_results,
        "accepted": compatible,
        "rejected": rejected
    }


# ============================================================================
# AUDIO SYNTHESIS
# ============================================================================

def concept_to_frequency(
    concept_name
):
    """
    Map concepts to simple synthesis frequencies.
    """

    frequency_map = {

        "Piano Music":
            220,

        "Audience Applause":
            330,

        "Car Engine":
            110,

        "Traffic Ambience":
            140,

        "Ocean Waves":
            80,

        "Bird Chirps":
            440,

        "Ambient Synth":
            260,

        "Soft Piano":
            196
    }

    return frequency_map.get(
        concept_name,
        220
    )


def generate_verified_audio(
    verified_concepts,
    output_path,
    sample_rate=22050,
    duration=3.0
):
    """
    Generate a lightweight prototype WAV.

    NOTE:
    This is synthetic demonstration audio,
    not a neural image-to-audio model output.
    """

    time_axis = np.linspace(
        0,
        duration,
        int(sample_rate * duration),
        endpoint=False
    )

    waveform = np.zeros_like(
        time_axis
    )

    for concept_data in verified_concepts:

        concept_name = concept_data["concept"]

        score = concept_data["score"]

        frequency = concept_to_frequency(
            concept_name
        )

        amplitude = (
            0.20
            + score * 0.30
        )

        waveform += (
            amplitude
            * np.sin(
                2
                * np.pi
                * frequency
                * time_axis
            )
        )

    if np.max(
        np.abs(waveform)
    ) > 0:

        waveform = (
            waveform
            / (
                np.max(
                    np.abs(waveform)
                )
                + 1e-8
            )
        )

    audio_int16 = (
        waveform
        * 32767
    ).astype(
        np.int16
    )

    wav_write(
        str(output_path),
        sample_rate,
        audio_int16
    )

    return output_path


# ============================================================================
# SAVE EXPERIMENT
# ============================================================================

def save_experiment(
    image_path,
    analysis,
    candidates,
    verification,
    audio_path=None
):
    """
    Save one complete reproducible experiment.
    """

    experiment_id = (
        datetime.now().strftime(
            "%Y%m%d_%H%M%S"
        )
        + "_"
        + uuid.uuid4().hex[:6]
    )

    experiment_path = (
        EXPERIMENT_DIR
        / experiment_id
    )

    experiment_path.mkdir(
        exist_ok=True
    )

    # ----------------------------------------------------------------------
    # COPY INPUT IMAGE
    # ----------------------------------------------------------------------

    image = Image.open(
        image_path
    )

    saved_image_path = (
        experiment_path
        / "input_image.png"
    )

    image.save(
        saved_image_path
    )

    # ----------------------------------------------------------------------
    # SAVE ANALYSIS
    # ----------------------------------------------------------------------

    with open(
        experiment_path / "analysis.json",
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(
            analysis,
            f,
            indent=4
        )

    # ----------------------------------------------------------------------
    # SAVE CANDIDATES
    # ----------------------------------------------------------------------

    with open(
        experiment_path / "candidates.json",
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(
            candidates,
            f,
            indent=4
        )

    # ----------------------------------------------------------------------
    # SAVE VERIFICATION
    # ----------------------------------------------------------------------

    with open(
        experiment_path / "verification.json",
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(
            verification,
            f,
            indent=4
        )

    # ----------------------------------------------------------------------
    # COPY AUDIO
    # ----------------------------------------------------------------------

    if (
        audio_path
        and os.path.exists(audio_path)
    ):

        with open(
            audio_path,
            "rb"
        ) as source:

            audio_data = source.read()

        with open(
            experiment_path / "verified_audio.wav",
            "wb"
        ) as destination:

            destination.write(
                audio_data
            )

    return experiment_path


# ============================================================================
# LOAD EXPERIMENTS
# ============================================================================

def get_experiments():

    experiments = []

    for path in sorted(
        EXPERIMENT_DIR.glob("*"),
        reverse=True
    ):

        if path.is_dir():

            experiments.append(
                path
            )

    return experiments


def load_json_file(path):

    try:

        with open(
            path,
            "r",
            encoding="utf-8"
        ) as f:

            return json.load(f)

    except Exception:

        return None


# ============================================================================
# APP HEADER
# ============================================================================

st.markdown(
    """
<div class="casv-hero">

<div class="casv-badge">
CAS-V RESEARCH PROTOTYPE
</div>

<div class="casv-title">
Conflict-Aware Semantic Verification
</div>

<div class="casv-subtitle">

A lightweight pre-generation verification layer for
image-to-audio research. The system analyzes visual characteristics,
evaluates candidate audio concepts, detects semantic conflicts,
and retains only concepts that pass the CAS-V verification stage.

</div>

</div>
""",
    unsafe_allow_html=True
)


# ============================================================================
# SIDEBAR
# ============================================================================

st.sidebar.markdown(
    "# 🎵 CAS-V"
)

st.sidebar.caption(
    "Research Prototype Interface"
)

page = st.sidebar.radio(
    "Navigation",
    [
        "🏠 Research Overview",
        "🎵 CAS-V Demo",
        "🧪 Experiments",
        "⚙️ Configuration"
    ],
    label_visibility="collapsed"
)

st.sidebar.markdown("---")

st.sidebar.caption(
    "Current implementation: "
    "visual-statistics heuristics + "
    "rule-based semantic verification + "
    "synthetic demonstration audio."
)


# ============================================================================
# PAGE 1 — RESEARCH OVERVIEW
# ============================================================================

if page == "🏠 Research Overview":

    st.markdown(
        "## Research Problem"
    )

    st.markdown(
        """
<div class="research-card">

<h4>Why CAS-V?</h4>

<p>

Image-to-audio systems may generate acoustically plausible outputs
without guaranteeing that the generated sound is semantically
consistent with the visual input.

</p>

<p>

CAS-V introduces a verification stage between
candidate audio selection and final audio synthesis.

</p>

</div>
""",
        unsafe_allow_html=True
    )

    # ----------------------------------------------------------------------
    # RESEARCH CONTRIBUTION
    # ----------------------------------------------------------------------

    col1, col2, col3 = st.columns(3)

    with col1:

        st.markdown(
            """
<div class="research-card">

<h4>🖼️ Visual Analysis</h4>

The prototype extracts basic visual characteristics and produces a
heuristic scene hypothesis.

</div>
""",
            unsafe_allow_html=True
        )

    with col2:

        st.markdown(
            """
<div class="research-card">

<h4>🔍 CAS-V Verification</h4>

Candidate audio concepts are evaluated against the inferred scene
category.

</div>
""",
            unsafe_allow_html=True
        )

    with col3:

        st.markdown(
            """
<div class="research-card">

<h4>🎵 Verified Output</h4>

Only concepts accepted by the verification stage are used for
prototype audio synthesis.

</div>
""",
            unsafe_allow_html=True
        )

    # ----------------------------------------------------------------------
    # PIPELINE
    # ----------------------------------------------------------------------

    st.markdown(
        "---"
    )

    st.markdown(
        "## CAS-V Pipeline"
    )

    pipeline = st.columns(
        5,
        gap="small"
    )

    pipeline_steps = [

        (
            "📤",
            "Input",
            "Upload an image"
        ),

        (
            "👁️",
            "Visual Analysis",
            "Extract visual statistics"
        ),

        (
            "🎼",
            "Candidates",
            "Generate possible audio concepts"
        ),

        (
            "🛡️",
            "CAS-V",
            "Detect semantic conflicts"
        ),

        (
            "🔊",
            "Output",
            "Generate verified prototype audio"
        )
    ]

    for column, data in zip(
        pipeline,
        pipeline_steps
    ):

        icon, title, description = data

        with column:

            st.markdown(
                f"""
<div class="pipeline-card">

<div class="pipeline-icon">
{icon}
</div>

<div class="pipeline-title">
{title}
</div>

<div class="pipeline-desc">
{description}
</div>

</div>
""",
                unsafe_allow_html=True
            )

    # ----------------------------------------------------------------------
    # CURRENT LIMITATION
    # ----------------------------------------------------------------------

    st.markdown(
        "---"
    )

    st.markdown(
        "## Current Prototype Scope"
    )

    st.info(
        """
The current version is a lightweight research prototype.
It does not perform real object detection or deep semantic understanding.
Scene labels are heuristic hypotheses derived from image statistics.
The CAS-V verifier is rule-based, and the audio output is synthetic
demonstration audio.

The interface intentionally exposes these limitations so that
the experimental demonstration does not overstate the system's
current capabilities.
"""
    )

    # ----------------------------------------------------------------------
    # EXPERIMENT SUMMARY
    # ----------------------------------------------------------------------

    st.markdown(
        "---"
    )

    experiments = get_experiments()

    total_experiments = len(
        experiments
    )

    total_accepted = 0

    total_rejected = 0

    for experiment in experiments:

        verification = load_json_file(
            experiment / "verification.json"
        )

        if verification:

            total_accepted += len(
                verification.get(
                    "accepted",
                    []
                )
            )

            total_rejected += len(
                verification.get(
                    "rejected",
                    []
                )
            )

    total_decisions = (
        total_accepted
        + total_rejected
    )

    filtering_rate = 0

    if total_decisions > 0:

        filtering_rate = (
            total_rejected
            / total_decisions
        ) * 100

    metric1, metric2, metric3 = st.columns(
        3
    )

    with metric1:

        st.metric(
            "Saved Experiments",
            total_experiments
        )

    with metric2:

        st.metric(
            "Accepted Concepts",
            total_accepted
        )

    with metric3:

        st.metric(
            "Conflict Filtering Rate",
            f"{filtering_rate:.1f}%"
        )


# ============================================================================
# PAGE 2 — CAS-V DEMO
# ============================================================================

elif page == "🎵 CAS-V Demo":

    st.markdown(
        "## Interactive CAS-V Demonstration"
    )

    st.caption(
        "Upload an image, run the visual analysis, evaluate candidate "
        "audio concepts, and inspect the CAS-V verification decisions."
    )

    # ----------------------------------------------------------------------
    # STEP 1 — UPLOAD
    # ----------------------------------------------------------------------

    st.markdown(
        "### ① Input Image"
    )

    uploaded_file = st.file_uploader(
        "Upload an image",
        type=[
            "jpg",
            "jpeg",
            "png"
        ],
        key="casv_demo_upload"
    )

    if uploaded_file is not None:

        image_path = save_uploaded_image(
            uploaded_file
        )

        st.session_state[
            "last_image_path"
        ] = str(
            image_path
        )

        col_image, col_info = st.columns(
            [1, 1],
            gap="large"
        )

        with col_image:

            st.image(
                Image.open(
                    image_path
                ),
                use_container_width=True,
                caption="Input Image"
            )

        with col_info:

            st.markdown(
                """
<div class="info-card">

<b>Research Note</b>

<br><br>

The image is processed using the current lightweight
visual-statistics module. The resulting scene label is
a heuristic hypothesis rather than object-level recognition.

</div>
""",
                unsafe_allow_html=True
            )

            run_analysis = st.button(
                "▶ Run CAS-V Analysis",
                use_container_width=True
            )

        # ------------------------------------------------------------------
        # RUN PIPELINE
        # ------------------------------------------------------------------

        if run_analysis:

            # --------------------------------------------------------------
            # STEP 2 — ANALYSIS
            # --------------------------------------------------------------

            analysis = analyze_uploaded_image(
                image_path
            )

            st.session_state[
                "last_analysis"
            ] = analysis

            st.markdown(
                "---"
            )

            st.markdown(
                "### ② Visual Analysis"
            )

            analysis_left, analysis_right = st.columns(
                [1.2, 1]
            )

            with analysis_left:

                st.markdown(
                    """
<div class="research-card">
""",
                    unsafe_allow_html=True
                )

                st.markdown(
                    "#### Visual Interpretation"
                )

                st.write(
                    f"**Scene Hypothesis:** "
                    f"{analysis['scene']}"
                )

                st.write(
                    f"**Confidence:** "
                    f"{analysis['confidence']}"
                )

                st.write(
                    f"**Dominant Color:** "
                    f"{analysis['dominant_color']}"
                )

                st.caption(
                    analysis[
                        "description"
                    ]
                )

                st.markdown(
                    "</div>",
                    unsafe_allow_html=True
                )

            with analysis_right:

                metric_a, metric_b = st.columns(
                    2
                )

                with metric_a:

                    st.metric(
                        "Brightness",
                        f"{analysis['brightness']:.3f}"
                    )

                with metric_b:

                    st.metric(
                        "Contrast",
                        f"{analysis['contrast']:.3f}"
                    )

                st.caption(
                    f"Method: "
                    f"{analysis['method']}"
                )

            # --------------------------------------------------------------
            # STEP 3 — CANDIDATES
            # --------------------------------------------------------------

            candidates = get_candidate_audio_concepts()

            st.session_state[
                "last_candidates"
            ] = candidates

            st.markdown(
                "---"
            )

            st.markdown(
                "### ③ Candidate Audio Concepts"
            )

            st.caption(
                "Candidates are intentionally presented without "
                "pre-assigned compatibility labels. "
                "CAS-V performs the accept/reject decision."
            )

            candidate_dataframe = pd.DataFrame(
                candidates
            )

            candidate_dataframe[
                "score"
            ] = (
                candidate_dataframe[
                    "score"
                ] * 100
            ).round(
                1
            )

            candidate_dataframe = (
                candidate_dataframe.rename(
                    columns={
                        "concept":
                            "Audio Concept",

                        "score":
                            "Candidate Score (%)"
                    }
                )
            )

            st.dataframe(
                candidate_dataframe,
                hide_index=True,
                use_container_width=True
            )

            # --------------------------------------------------------------
            # STEP 4 — CAS-V
            # --------------------------------------------------------------

            verification = semantic_verifier(
                analysis,
                candidates
            )

            st.session_state[
                "last_verification"
            ] = verification

            st.markdown(
                "---"
            )

            st.markdown(
                "### ④ CAS-V Semantic Verification"
            )

            accepted = verification[
                "accepted"
            ]

            rejected = verification[
                "rejected"
            ]

            total_candidates = len(
                verification[
                    "results"
                ]
            )

            filtering_rate = (
                len(rejected)
                / total_candidates
            ) * 100

            st.markdown(
                f"""
<div class="verification-summary">

<div class="verification-title">

🛡️ CAS-V Decision Summary

</div>

<b>Heuristic Scene Category:</b>
{verification['scene_category'].upper()}

<br><br>

<b>Candidate Concepts:</b>
{total_candidates}

&nbsp;&nbsp; • &nbsp;&nbsp;

<b>Accepted:</b>
{len(accepted)}

&nbsp;&nbsp; • &nbsp;&nbsp;

<b>Rejected:</b>
{len(rejected)}

&nbsp;&nbsp; • &nbsp;&nbsp;

<b>Conflict Filtering:</b>
{filtering_rate:.1f}%

</div>
""",
                unsafe_allow_html=True
            )

            # --------------------------------------------------------------
            # DECISION TABLE
            # --------------------------------------------------------------

            verification_rows = []

            for result in verification[
                "results"
            ]:

                verification_rows.append(
                    {
                        "Audio Concept":
                            result["concept"],

                        "Candidate Score":
                            f"{result['score'] * 100:.1f}%",

                        "Concept Category":
                            result["concept_category"],

                        "CAS-V Decision":
                            (
                                "✓ ACCEPT"
                                if result["decision"]
                                == "Accepted"
                                else "✗ REJECT"
                            )
                    }
                )

            verification_df = pd.DataFrame(
                verification_rows
            )

            st.dataframe(
                verification_df,
                hide_index=True,
                use_container_width=True
            )

            # --------------------------------------------------------------
            # DETAILED REASONS
            # --------------------------------------------------------------

            st.markdown(
                "#### Verification Explanations"
            )

            accept_col, reject_col = st.columns(
                2,
                gap="large"
            )

            with accept_col:

                st.markdown(
                    "##### ✓ Accepted"
                )

                if accepted:

                    for item in accepted:

                        st.markdown(
                            f"""
<div class="accept-card">

<b>✓ {item['concept']}</b>

<br><br>

{item['reason']}

</div>
""",
                            unsafe_allow_html=True
                        )

                else:

                    st.info(
                        "No concepts were accepted."
                    )

            with reject_col:

                st.markdown(
                    "##### ✗ Rejected"
                )

                if rejected:

                    for item in rejected:

                        st.markdown(
                            f"""
<div class="reject-card">

<b>✗ {item['concept']}</b>

<br><br>

{item['reason']}

</div>
""",
                            unsafe_allow_html=True
                        )

                else:

                    st.success(
                        "No semantic conflicts were detected."
                    )

            # --------------------------------------------------------------
            # STEP 5 — AUDIO OUTPUT
            # --------------------------------------------------------------

            st.markdown(
                "---"
            )

            st.markdown(
                "### ⑤ Verified Audio Output"
            )

            if accepted:

                temporary_audio_path = (
                    BASE_DIR
                    / "current_verified_audio.wav"
                )

                generate_verified_audio(
                    accepted,
                    temporary_audio_path,
                    sample_rate=
                        st.session_state[
                            "sample_rate"
                        ],
                    duration=
                        st.session_state[
                            "audio_duration"
                        ]
                )

                st.session_state[
                    "last_audio_path"
                ] = str(
                    temporary_audio_path
                )

                st.success(
                    f"CAS-V retained "
                    f"{len(accepted)} compatible "
                    f"concept(s) and filtered "
                    f"{len(rejected)} conflict(s)."
                )

                st.audio(
                    str(
                        temporary_audio_path
                    ),
                    format="audio/wav"
                )

                audio_col1, audio_col2 = st.columns(
                    2
                )

                with audio_col1:

                    with open(
                        temporary_audio_path,
                        "rb"
                    ) as f:

                        st.download_button(
                            "📥 Download Verified Audio",
                            f.read(),
                            file_name=
                                "casv_verified_audio.wav",
                            mime=
                                "audio/wav",
                            use_container_width=True
                        )

                with audio_col2:

                    if st.button(
                        "💾 Save Complete Experiment",
                        use_container_width=True
                    ):

                        experiment_path = (
                            save_experiment(
                                image_path,
                                analysis,
                                candidates,
                                verification,
                                temporary_audio_path
                            )
                        )

                        st.success(
                            "Experiment saved successfully: "
                            f"{experiment_path.name}"
                        )

            else:

                st.warning(
                    """
No candidate concept passed the CAS-V verification stage.
Prototype audio synthesis was therefore not performed.
"""
                )

    else:

        st.markdown(
            """
<div class="research-card">

<h4>📤 Ready to Start an Experiment</h4>

Upload an image to begin the CAS-V workflow.

The system will:

<ol>

<li>Extract visual statistics</li>

<li>Generate a heuristic scene hypothesis</li>

<li>Evaluate candidate audio concepts</li>

<li>Apply CAS-V verification</li>

<li>Generate synthetic prototype audio from accepted concepts</li>

</ol>

</div>
""",
            unsafe_allow_html=True
        )


# ============================================================================
# PAGE 3 — EXPERIMENTS
# ============================================================================

elif page == "🧪 Experiments":

    st.markdown(
        "## Experiment Dashboard"
    )

    experiments = get_experiments()

    # ----------------------------------------------------------------------
    # SUMMARY
    # ----------------------------------------------------------------------

    total_experiments = len(
        experiments
    )

    total_audio = len(
        [
            e for e in experiments
            if (
                e
                / "verified_audio.wav"
            ).exists()
        ]
    )

    total_candidates = 0

    total_rejected = 0

    for experiment in experiments:

        verification = load_json_file(
            experiment
            / "verification.json"
        )

        if verification:

            results = verification.get(
                "results",
                []
            )

            rejected = verification.get(
                "rejected",
                []
            )

            total_candidates += len(
                results
            )

            total_rejected += len(
                rejected
            )

    overall_filter_rate = 0

    if total_candidates > 0:

        overall_filter_rate = (
            total_rejected
            / total_candidates
        ) * 100

    m1, m2, m3 = st.columns(
        3
    )

    with m1:

        st.metric(
            "Saved Experiments",
            total_experiments
        )

    with m2:

        st.metric(
            "Verified Audio Outputs",
            total_audio
        )

    with m3:

        st.metric(
            "Overall Conflict Filtering",
            f"{overall_filter_rate:.1f}%"
        )

    st.markdown(
        "---"
    )

    # ----------------------------------------------------------------------
    # NO EXPERIMENTS
    # ----------------------------------------------------------------------

    if not experiments:

        st.info(
            """
No saved experiments yet.

Go to the CAS-V Demo page, run an experiment,
and click "Save Complete Experiment".
"""
        )

    else:

        experiment_names = [
            experiment.name
            for experiment in experiments
        ]

        selected_name = st.selectbox(
            "Select an experiment",
            experiment_names
        )

        selected_experiment = (
            EXPERIMENT_DIR
            / selected_name
        )

        image_path = (
            selected_experiment
            / "input_image.png"
        )

        analysis = load_json_file(
            selected_experiment
            / "analysis.json"
        )

        candidates = load_json_file(
            selected_experiment
            / "candidates.json"
        )

        verification = load_json_file(
            selected_experiment
            / "verification.json"
        )

        audio_path = (
            selected_experiment
            / "verified_audio.wav"
        )

        # ------------------------------------------------------------------
        # EXPERIMENT IMAGE + ANALYSIS
        # ------------------------------------------------------------------

        st.markdown(
            "### Selected Experiment"
        )

        exp_col1, exp_col2 = st.columns(
            [1, 1],
            gap="large"
        )

        with exp_col1:

            if image_path.exists():

                st.image(
                    Image.open(
                        image_path
                    ),
                    use_container_width=True,
                    caption=
                        "Experiment Input Image"
                )

        with exp_col2:

            if analysis:

                st.markdown(
                    "#### Visual Analysis"
                )

                st.write(
                    f"**Scene Hypothesis:** "
                    f"{analysis['scene']}"
                )

                st.write(
                    f"**Confidence:** "
                    f"{analysis['confidence']}"
                )

                st.write(
                    f"**Dominant Color:** "
                    f"{analysis['dominant_color']}"
                )

                metric_x, metric_y = st.columns(
                    2
                )

                with metric_x:

                    st.metric(
                        "Brightness",
                        f"{analysis['brightness']:.3f}"
                    )

                with metric_y:

                    st.metric(
                        "Contrast",
                        f"{analysis['contrast']:.3f}"
                    )

                st.caption(
                    analysis["description"]
                )

        # ------------------------------------------------------------------
        # VERIFICATION RESULTS
        # ------------------------------------------------------------------

        if verification:

            st.markdown(
                "---"
            )

            st.markdown(
                "### CAS-V Verification Results"
            )

            accepted = verification.get(
                "accepted",
                []
            )

            rejected = verification.get(
                "rejected",
                []
            )

            result_rows = []

            for result in verification.get(
                "results",
                []
            ):

                result_rows.append(
                    {
                        "Concept":
                            result["concept"],

                        "Score":
                            f"{result['score'] * 100:.1f}%",

                        "Category":
                            result["concept_category"],

                        "Decision":
                            result["decision"]
                    }
                )

            st.dataframe(
                pd.DataFrame(
                    result_rows
                ),
                hide_index=True,
                use_container_width=True
            )

            total = (
                len(accepted)
                + len(rejected)
            )

            if total > 0:

                filter_rate = (
                    len(rejected)
                    / total
                ) * 100

            else:

                filter_rate = 0

            impact1, impact2, impact3 = st.columns(
                3
            )

            with impact1:

                st.metric(
                    "Accepted",
                    len(accepted)
                )

            with impact2:

                st.metric(
                    "Rejected",
                    len(rejected)
                )

            with impact3:

                st.metric(
                    "Conflict Filtering",
                    f"{filter_rate:.1f}%"
                )

        # ------------------------------------------------------------------
        # AUDIO
        # ------------------------------------------------------------------

        st.markdown(
            "---"
        )

        st.markdown(
            "### Verified Audio"
        )

        if audio_path.exists():

            st.audio(
                str(
                    audio_path
                ),
                format="audio/wav"
            )

            with open(
                audio_path,
                "rb"
            ) as f:

                st.download_button(
                    "📥 Download Experiment Audio",
                    f.read(),
                    file_name=
                        f"{selected_name}_verified_audio.wav",
                    mime=
                        "audio/wav"
                )

        else:

            st.warning(
                "No verified audio was saved for this experiment."
            )

        # ------------------------------------------------------------------
        # EXPORT JSON
        # ------------------------------------------------------------------

        st.markdown(
            "---"
        )

        st.markdown(
            "### Experiment Data Export"
        )

        export_data = {
            "experiment_id":
                selected_name,

            "analysis":
                analysis,

            "candidates":
                candidates,

            "verification":
                verification
        }

        export_json = json.dumps(
            export_data,
            indent=4
        )

        st.download_button(
            "📊 Download Experiment JSON",
            export_json,
            file_name=
                f"{selected_name}_results.json",
            mime=
                "application/json"
        )


# ============================================================================
# PAGE 4 — CONFIGURATION
# ============================================================================

elif page == "⚙️ Configuration":

    st.markdown(
        "## Prototype Configuration"
    )

    st.caption(
        "These parameters directly affect the prototype audio output."
    )

    config_col1, config_col2 = st.columns(
        2
    )

    with config_col1:

        st.markdown(
            "### Audio Parameters"
        )

        sample_rate = st.selectbox(
            "Sample Rate",
            [
                16000,
                22050,
                44100,
                48000
            ],
            index=[
                16000,
                22050,
                44100,
                48000
            ].index(
                st.session_state[
                    "sample_rate"
                ]
            )
        )

        audio_duration = st.slider(
            "Audio Duration (seconds)",
            min_value=1.0,
            max_value=10.0,
            value=float(
                st.session_state[
                    "audio_duration"
                ]
            ),
            step=0.5
        )

        if st.button(
            "Save Audio Configuration"
        ):

            st.session_state[
                "sample_rate"
            ] = sample_rate

            st.session_state[
                "audio_duration"
            ] = audio_duration

            st.success(
                "Audio configuration updated."
            )

    with config_col2:

        st.markdown(
            "### Current CAS-V Prototype"

        )

        st.markdown(
            """
<div class="research-card">

<b>Visual Module</b>

<br>

Visual statistics:
brightness, contrast, average RGB.

<br><br>

<b>Scene Interpretation</b>

<br>

Rule-based heuristic scene hypothesis.

<br><br>

<b>Verifier</b>

<br>

Rule-based compatibility comparison between
scene categories and audio concept categories.

<br><br>

<b>Audio Output</b>

<br>

Simple sinusoidal demonstration synthesis.

</div>
""",
            unsafe_allow_html=True
        )

    st.markdown(
        "---"
    )

    st.markdown(
        "### Workspace Information"
    )

    info1, info2, info3 = st.columns(
        3
    )

    with info1:

        st.metric(
            "Uploaded Images",
            len(
                list(
                    UPLOAD_DIR.glob("*")
                )
            )
        )

    with info2:

        st.metric(
            "Saved Experiments",
            len(
                get_experiments()
            )
        )

    with info3:

        st.metric(
            "Python",
            sys.version.split()[0]
        )

    st.markdown(
        "---"
    )

    st.markdown(
        "### Cleanup"
    )

    st.warning(
        "Deleting experiments cannot be undone."
    )

    if st.button(
        "🗑️ Delete All Saved Experiments"
    ):

        deleted_count = 0

        for experiment in get_experiments():

            for file in experiment.glob(
                "*"
            ):

                try:

                    file.unlink()

                except Exception:

                    pass

            try:

                experiment.rmdir()

                deleted_count += 1

            except Exception:

                pass

        st.success(
            f"Deleted {deleted_count} experiment(s)."
        )


# ============================================================================
# FOOTER
# ============================================================================

st.markdown(
    "---"
)

st.markdown(
    """
<div style="
    text-align:center;
    color:#64748b;
    font-size:0.85rem;
    padding:1rem;
">

<b>CAS-V Research Prototype</b>

<br>

Conflict-Aware Semantic Verification for Reliable Image-to-Audio Research

<br><br>

Current prototype:
Visual heuristics → Candidate concepts → CAS-V verification →
Synthetic demonstration audio

</div>
""",
    unsafe_allow_html=True
)