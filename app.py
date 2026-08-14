# =============================================================================
# CAS-V 2.0
# Conflict-Aware Semantic Verification for Reliable Image-to-Audio Generation
#
# Research Prototype
#
# Pipeline:
#
# Image
#   ↓
# Multi-Signal Visual Analysis
#   ↓
# Structured Semantic Profile
#   ↓
# Audio Candidate Pool
#   ↓
# Multi-Dimensional CAS-V Verification
#   ├── Scene Compatibility
#   ├── Object Compatibility
#   ├── Environment Compatibility
#   ├── Mood Compatibility
#   ├── Semantic Similarity
#   ├── Explicit Conflict Detection
#   └── Uncertainty / Resilience Analysis
#   ↓
# ACCEPT / REPAIR / REJECT / UNCERTAIN
#   ↓
# Verified Prototype Audio
#   ↓
# Experiment Storage
# =============================================================================


import streamlit as st
import os
import json
import uuid
import shutil
import sys

from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd

from PIL import Image
from scipy.io.wavfile import write as wav_write


# =============================================================================
# 1. PAGE CONFIGURATION
# =============================================================================

st.set_page_config(
    page_title="CAS-V 2.0 Research Prototype",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)


# =============================================================================
# 2. PROJECT DIRECTORIES
# =============================================================================

BASE_DIR = Path.cwd()

UPLOAD_DIR = BASE_DIR / "uploaded_images"
EXPERIMENT_DIR = BASE_DIR / "experiments"
TEMP_DIR = BASE_DIR / "temporary"

UPLOAD_DIR.mkdir(exist_ok=True)
EXPERIMENT_DIR.mkdir(exist_ok=True)
TEMP_DIR.mkdir(exist_ok=True)


# =============================================================================
# 3. SESSION STATE
# =============================================================================

DEFAULTS = {

    "sample_rate": 22050,

    "audio_duration": 4.0,

    "accept_threshold": 0.75,

    "repair_threshold": 0.45,

    "uncertainty_threshold": 0.40,

    "last_image_path": None,

    "last_analysis": None,

    "last_semantic_profile": None,

    "last_candidates": None,

    "last_verification": None,

    "last_audio_path": None,

    "last_experiment_id": None
}


for key, value in DEFAULTS.items():

    if key not in st.session_state:

        st.session_state[key] = value


# =============================================================================
# 4. CUSTOM CSS
# =============================================================================

st.markdown(
    """
<style>

/* ============================================================================
   GLOBAL
============================================================================ */

.stApp {

    background:
        radial-gradient(
            circle at top left,
            rgba(37, 99, 235, 0.10),
            transparent 28%
        ),

        radial-gradient(
            circle at top right,
            rgba(124, 58, 237, 0.08),
            transparent 30%
        ),

        #f8fafc;
}


.main .block-container {

    max-width: 1500px;

    padding-top: 2rem;

    padding-bottom: 3rem;
}


/* ============================================================================
   HERO
============================================================================ */

.casv-hero {

    background:
        linear-gradient(
            135deg,
            rgba(255,255,255,0.98),
            rgba(239,246,255,0.96)
        );

    border:
        1px solid rgba(148,163,184,0.20);

    border-radius:
        26px;

    padding:
        2.4rem;

    margin-bottom:
        2rem;

    box-shadow:
        0 20px 45px rgba(15,23,42,0.07);
}


.casv-badge {

    display:
        inline-block;

    background:
        linear-gradient(
            135deg,
            #2563eb,
            #7c3aed
        );

    color:
        white;

    border-radius:
        999px;

    padding:
        0.45rem 1rem;

    font-size:
        0.72rem;

    font-weight:
        800;

    letter-spacing:
        0.12em;

    margin-bottom:
        0.9rem;
}


.casv-title {

    font-size:
        2.6rem;

    font-weight:
        900;

    margin:
        0;

    background:
        linear-gradient(
            135deg,
            #1e3a8a,
            #7c3aed
        );

    -webkit-background-clip:
        text;

    -webkit-text-fill-color:
        transparent;
}


.casv-subtitle {

    color:
        #64748b;

    margin-top:
        0.8rem;

    max-width:
        950px;

    line-height:
        1.7;
}


/* ============================================================================
   GENERAL CARDS
============================================================================ */

.research-card {

    background:
        rgba(255,255,255,0.92);

    border:
        1px solid rgba(148,163,184,0.18);

    border-radius:
        18px;

    padding:
        1.4rem;

    margin-bottom:
        1rem;

    box-shadow:
        0 8px 24px rgba(15,23,42,0.04);
}


.info-card {

    background:
        rgba(239,246,255,0.78);

    border-left:
        5px solid #2563eb;

    border-radius:
        12px;

    padding:
        1rem;

    margin-bottom:
        1rem;
}


.accept-card {

    background:
        rgba(236,253,245,0.82);

    border-left:
        5px solid #10b981;

    border-radius:
        12px;

    padding:
        1rem;

    margin-bottom:
        0.8rem;
}


.repair-card {

    background:
        rgba(255,251,235,0.90);

    border-left:
        5px solid #f59e0b;

    border-radius:
        12px;

    padding:
        1rem;

    margin-bottom:
        0.8rem;
}


.reject-card {

    background:
        rgba(254,242,242,0.85);

    border-left:
        5px solid #ef4444;

    border-radius:
        12px;

    padding:
        1rem;

    margin-bottom:
        0.8rem;
}


.uncertain-card {

    background:
        rgba(243,244,246,0.90);

    border-left:
        5px solid #64748b;

    border-radius:
        12px;

    padding:
        1rem;

    margin-bottom:
        0.8rem;
}


/* ============================================================================
   PIPELINE
============================================================================ */

.pipeline-card {

    text-align:
        center;

    background:
        rgba(255,255,255,0.94);

    border:
        1px solid rgba(148,163,184,0.18);

    border-radius:
        18px;

    padding:
        1.2rem;

    min-height:
        180px;
}


.pipeline-icon {

    font-size:
        2rem;

    margin-bottom:
        0.6rem;
}


.pipeline-title {

    font-weight:
        850;

    margin-bottom:
        0.5rem;
}


.pipeline-desc {

    color:
        #64748b;

    font-size:
        0.82rem;

    line-height:
        1.5;
}


/* ============================================================================
   VERIFICATION SUMMARY
============================================================================ */

.verification-summary {

    background:
        linear-gradient(
            135deg,
            rgba(37,99,235,0.08),
            rgba(124,58,237,0.08)
        );

    border:
        1px solid rgba(99,102,241,0.20);

    border-radius:
        18px;

    padding:
        1.5rem;

    margin:
        1rem 0 1.5rem 0;
}


.verification-title {

    font-weight:
        900;

    font-size:
        1.25rem;

    color:
        #1e3a8a;

    margin-bottom:
        0.8rem;
}


/* ============================================================================
   SCORE BARS
============================================================================ */

.score-label {

    font-size:
        0.82rem;

    font-weight:
        700;

    color:
        #334155;
}


.score-container {

    background:
        #e2e8f0;

    border-radius:
        999px;

    overflow:
        hidden;

    height:
        10px;

    margin-top:
        0.35rem;

    margin-bottom:
        0.7rem;
}


.score-fill {

    height:
        100%;

    background:
        linear-gradient(
            90deg,
            #2563eb,
            #7c3aed
        );

    border-radius:
        999px;
}


/* ============================================================================
   METRICS
============================================================================ */

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


/* ============================================================================
   BUTTONS
============================================================================ */

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
        750;

    padding:
        0.65rem 1rem;
}


.stButton > button:hover {

    transform:
        translateY(-1px);
}


/* ============================================================================
   FILE UPLOADER
============================================================================ */

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


/* ============================================================================
   SIDEBAR
============================================================================ */

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


# =============================================================================
# 5. HELPER FUNCTIONS
# =============================================================================

def clamp(value, minimum=0.0, maximum=1.0):

    return max(
        minimum,
        min(value, maximum)
    )


def safe_filename(name):

    return "".join(

        character

        if (
            character.isalnum()
            or character in "._-"
        )

        else "_"

        for character in name
    )


def save_uploaded_image(uploaded_file):

    timestamp = datetime.now().strftime(
        "%Y%m%d_%H%M%S"
    )

    unique_id = uuid.uuid4().hex[:8]

    filename = (
        f"{timestamp}_"
        f"{unique_id}_"
        f"{safe_filename(uploaded_file.name)}"
    )

    image_path = (
        UPLOAD_DIR
        / filename
    )

    with open(
        image_path,
        "wb"
    ) as file:

        file.write(
            uploaded_file.getbuffer()
        )

    return image_path


# =============================================================================
# 6. MULTI-SIGNAL VISUAL ANALYSIS
# =============================================================================

def analyze_uploaded_image(image_path):

    """
    Multi-signal visual analysis.

    This lightweight prototype calculates:

    - Brightness
    - Contrast
    - Color distribution
    - Color dominance
    - Warm/cool ratio
    - Visual complexity

    IMPORTANT:

    This module is heuristic.

    It does not claim to perform true object detection.
    """

    image = Image.open(
        image_path
    ).convert(
        "RGB"
    )

    array = (
        np.array(image)
        .astype(np.float32)
        / 255.0
    )

    # -------------------------------------------------------------------------
    # BASIC SIGNALS
    # -------------------------------------------------------------------------

    brightness = float(
        array.mean()
    )

    contrast = float(
        array.std()
    )

    red_mean = float(
        array[:, :, 0].mean()
    )

    green_mean = float(
        array[:, :, 1].mean()
    )

    blue_mean = float(
        array[:, :, 2].mean()
    )

    color_values = {

        "red":
            red_mean,

        "green":
            green_mean,

        "blue":
            blue_mean
    }

    dominant_color = max(
        color_values,
        key=color_values.get
    )

    # -------------------------------------------------------------------------
    # COLOR RATIOS
    # -------------------------------------------------------------------------

    warm_signal = (
        red_mean
        + 0.5 * green_mean
    ) / 1.5

    cool_signal = (
        blue_mean
        + 0.5 * green_mean
    ) / 1.5

    warm_ratio = clamp(
        warm_signal
    )

    cool_ratio = clamp(
        cool_signal
    )

    # -------------------------------------------------------------------------
    # VISUAL COMPLEXITY
    # -------------------------------------------------------------------------

    complexity = clamp(
        contrast * 2.5
    )

    # -------------------------------------------------------------------------
    # SCENE HYPOTHESIS
    # -------------------------------------------------------------------------

    scene_candidates = []

    if brightness < 0.35:

        scene_candidates.append({

            "label":
                "dim / low-light environment",

            "score":
                clamp(
                    (0.50 - brightness)
                    * 1.5
                )
        })

    if brightness > 0.60:

        scene_candidates.append({

            "label":
                "bright / open environment",

            "score":
                clamp(
                    brightness
                )
        })

    if cool_ratio > warm_ratio + 0.08:

        scene_candidates.append({

            "label":
                "cool-toned natural or open environment",

            "score":
                clamp(
                    cool_ratio
                )
        })

    if warm_ratio > cool_ratio + 0.08:

        scene_candidates.append({

            "label":
                "warm-toned environment",

            "score":
                clamp(
                    warm_ratio
                )
        })

    if complexity > 0.55:

        scene_candidates.append({

            "label":
                "visually complex environment",

            "score":
                complexity
        })

    if not scene_candidates:

        scene_candidates.append({

            "label":
                "general visual environment",

            "score":
                0.45
        })

    scene_candidates = sorted(

        scene_candidates,

        key=lambda item:
            item["score"],

        reverse=True
    )

    primary_scene = (
        scene_candidates[0]
    )

    # -------------------------------------------------------------------------
    # ENVIRONMENT HYPOTHESIS
    # -------------------------------------------------------------------------

    if brightness > 0.60:

        environment = "open / illuminated"

        environment_confidence = clamp(
            brightness
        )

    elif brightness < 0.35:

        environment = "dark / enclosed"

        environment_confidence = clamp(
            1.0 - brightness
        )

    else:

        environment = "mixed / uncertain"

        environment_confidence = 0.50

    # -------------------------------------------------------------------------
    # MOOD ESTIMATION
    # -------------------------------------------------------------------------

    # Valence:
    # Warm + brightness generally increases positive visual signal.

    valence = clamp(

        0.45
        + (
            warm_ratio
            * 0.30
        )
        + (
            brightness
            * 0.25
        )
        - (
            contrast
            * 0.10
        )
    )

    # Arousal:
    # Higher contrast and complexity increase estimated intensity.

    arousal = clamp(

        0.25
        + (
            contrast
            * 0.60
        )
        + (
            complexity
            * 0.25
        )
    )

    # -------------------------------------------------------------------------
    # MOOD LABEL
    # -------------------------------------------------------------------------

    if (
        valence >= 0.65
        and arousal < 0.55
    ):

        mood_label = "calm / positive"

    elif (
        arousal >= 0.65
    ):

        mood_label = "energetic / intense"

    elif (
        valence < 0.40
    ):

        mood_label = "dark / subdued"

    else:

        mood_label = "neutral / mixed"

    # -------------------------------------------------------------------------
    # SEMANTIC CONFIDENCE
    # -------------------------------------------------------------------------

    top_score = (
        primary_scene["score"]
    )

    second_score = 0.0

    if len(scene_candidates) > 1:

        second_score = (
            scene_candidates[1]["score"]
        )

    separation = abs(
        top_score
        - second_score
    )

    semantic_confidence = clamp(

        (
            top_score
            * 0.65
        )
        + (
            separation
            * 0.35
        )
    )

    return {

        "visual_features": {

            "brightness":
                brightness,

            "contrast":
                contrast,

            "complexity":
                complexity,

            "dominant_color":
                dominant_color,

            "red_mean":
                red_mean,

            "green_mean":
                green_mean,

            "blue_mean":
                blue_mean,

            "warm_ratio":
                warm_ratio,

            "cool_ratio":
                cool_ratio
        },

        "scene_candidates":
            scene_candidates,

        "primary_scene":
            primary_scene,

        "environment": {

            "label":
                environment,

            "confidence":
                environment_confidence
        },

        "mood": {

            "valence":
                valence,

            "arousal":
                arousal,

            "label":
                mood_label
        },

        "semantic_confidence":
            semantic_confidence,

        "analysis_method":
            (
                "Multi-signal visual statistics "
                "and heuristic semantic profiling"
            )
    }


# =============================================================================
# 7. STRUCTURED SEMANTIC PROFILE
# =============================================================================

def build_semantic_profile(analysis):

    """
    Convert raw visual analysis into a structured semantic profile.
    """

    scene_label = (
        analysis[
            "primary_scene"
        ][
            "label"
        ]
    )

    scene_confidence = (
        analysis[
            "primary_scene"
        ][
            "score"
        ]
    )

    environment = (
        analysis[
            "environment"
        ]
    )

    mood = (
        analysis[
            "mood"
        ]
    )

    # -------------------------------------------------------------------------
    # ABSTRACT CONCEPT EXTRACTION
    #
    # These are NOT object detections.
    # They are semantic hypotheses derived from visual signals.
    # -------------------------------------------------------------------------

    concepts = []

    scene_lower = (
        scene_label.lower()
    )

    if (
        "cool"
        in scene_lower
    ):

        concepts.extend([
            "open",
            "cool-toned",
            "possible nature"
        ])

    if (
        "warm"
        in scene_lower
    ):

        concepts.extend([
            "warm-toned",
            "possible activity"
        ])

    if (
        "bright"
        in scene_lower
    ):

        concepts.extend([
            "illuminated",
            "open"
        ])

    if (
        "dim"
        in scene_lower
    ):

        concepts.extend([
            "low-light",
            "enclosed"
        ])

    if (
        analysis[
            "visual_features"
        ][
            "complexity"
        ]
        > 0.55
    ):

        concepts.append(
            "complex visual structure"
        )

    concepts = list(
        dict.fromkeys(
            concepts
        )
    )

    return {

        "scene": {

            "label":
                scene_label,

            "confidence":
                scene_confidence
        },

        "environment":
            environment,

        "mood":
            mood,

        "semantic_concepts":
            concepts,

        "semantic_confidence":
            analysis[
                "semantic_confidence"
            ],

        "source_method":
            analysis[
                "analysis_method"
            ]
    }


# =============================================================================
# 8. AUDIO CANDIDATE KNOWLEDGE BASE
# =============================================================================

AUDIO_CANDIDATES = [

    {
        "concept":
            "Ocean Waves",

        "category":
            "natural",

        "scene_tags":
            [
                "open",
                "cool-toned",
                "nature"
            ],

        "environment_tags":
            [
                "open",
                "outdoor"
            ],

        "mood_profile":
            {
                "valence": 0.70,
                "arousal": 0.30
            }
    },

    {
        "concept":
            "Bird Chirps",

        "category":
            "natural",

        "scene_tags":
            [
                "open",
                "nature",
                "illuminated"
            ],

        "environment_tags":
            [
                "open",
                "outdoor"
            ],

        "mood_profile":
            {
                "valence": 0.80,
                "arousal": 0.40
            }
    },

    {
        "concept":
            "Forest Ambience",

        "category":
            "natural",

        "scene_tags":
            [
                "nature",
                "open"
            ],

        "environment_tags":
            [
                "outdoor"
            ],

        "mood_profile":
            {
                "valence": 0.65,
                "arousal": 0.35
            }
    },

    {
        "concept":
            "Piano Music",

        "category":
            "music",

        "scene_tags":
            [
                "indoor",
                "performance"
            ],

        "environment_tags":
            [
                "enclosed",
                "mixed"
            ],

        "mood_profile":
            {
                "valence": 0.65,
                "arousal": 0.40
            }
    },

    {
        "concept":
            "Soft Piano",

        "category":
            "ambient",

        "scene_tags":
            [
                "calm",
                "neutral"
            ],

        "environment_tags":
            [
                "mixed"
            ],

        "mood_profile":
            {
                "valence": 0.65,
                "arousal": 0.25
            }
    },

    {
        "concept":
            "Ambient Synth",

        "category":
            "ambient",

        "scene_tags":
            [
                "abstract",
                "mixed"
            ],

        "environment_tags":
            [
                "mixed"
            ],

        "mood_profile":
            {
                "valence": 0.55,
                "arousal": 0.35
            }
    },

    {
        "concept":
            "Car Engine",

        "category":
            "urban",

        "scene_tags":
            [
                "urban",
                "warm-toned",
                "complex"
            ],

        "environment_tags":
            [
                "open",
                "mixed"
            ],

        "mood_profile":
            {
                "valence": 0.45,
                "arousal": 0.70
            }
    },

    {
        "concept":
            "Traffic Ambience",

        "category":
            "urban",

        "scene_tags":
            [
                "urban",
                "complex"
            ],

        "environment_tags":
            [
                "open",
                "mixed"
            ],

        "mood_profile":
            {
                "valence": 0.40,
                "arousal": 0.75
            }
    },

    {
        "concept":
            "Audience Applause",

        "category":
            "performance",

        "scene_tags":
            [
                "performance",
                "complex"
            ],

        "environment_tags":
            [
                "enclosed",
                "mixed"
            ],

        "mood_profile":
            {
                "valence": 0.75,
                "arousal": 0.85
            }
    },

    {
        "concept":
            "Heavy Concert",

        "category":
            "performance",

        "scene_tags":
            [
                "performance",
                "dark",
                "complex"
            ],

        "environment_tags":
            [
                "enclosed"
            ],

        "mood_profile":
            {
                "valence": 0.45,
                "arousal": 0.95
            }
    }
]


# =============================================================================
# 9. CANDIDATE GENERATION
# =============================================================================

def generate_audio_candidates(
    semantic_profile,
    top_k=8
):

    """
    Generate candidates independently of the final CAS-V decision.

    The generator produces a broad candidate pool.
    It does NOT assign ACCEPT / REJECT labels.
    """

    candidates = []

    scene_text = (
        semantic_profile[
            "scene"
        ][
            "label"
        ].lower()
    )

    mood = (
        semantic_profile[
            "mood"
        ]
    )

    concepts = (
        semantic_profile[
            "semantic_concepts"
        ]
    )

    for item in AUDIO_CANDIDATES:

        relevance = 0.35

        # ---------------------------------------------------------------------
        # SIMPLE PRELIMINARY RELEVANCE
        # ---------------------------------------------------------------------

        for tag in item["scene_tags"]:

            if tag in scene_text:

                relevance += 0.12

            for concept in concepts:

                if tag in concept:

                    relevance += 0.08

        candidate_mood = (
            item[
                "mood_profile"
            ]
        )

        mood_distance = (

            abs(
                mood["valence"]
                - candidate_mood["valence"]
            )

            +

            abs(
                mood["arousal"]
                - candidate_mood["arousal"]
            )
        ) / 2

        mood_similarity = (
            1.0
            - mood_distance
        )

        relevance += (
            mood_similarity
            * 0.20
        )

        relevance = clamp(
            relevance
        )

        candidates.append({

            "concept":
                item["concept"],

            "category":
                item["category"],

            "preliminary_score":
                relevance,

            "scene_tags":
                item["scene_tags"],

            "environment_tags":
                item["environment_tags"],

            "mood_profile":
                item["mood_profile"]
        })

    candidates = sorted(

        candidates,

        key=lambda item:
            item["preliminary_score"],

        reverse=True
    )

    return candidates[:top_k]


# =============================================================================
# 10. SCENE COMPATIBILITY
# =============================================================================

def calculate_scene_compatibility(
    semantic_profile,
    candidate
):

    scene_text = (
        semantic_profile[
            "scene"
        ][
            "label"
        ].lower()
    )

    semantic_concepts = (
        semantic_profile[
            "semantic_concepts"
        ]
    )

    matches = 0

    possible = max(
        1,
        len(candidate["scene_tags"])
    )

    for tag in candidate["scene_tags"]:

        if tag in scene_text:

            matches += 1

            continue

        for concept in semantic_concepts:

            if tag in concept:

                matches += 1

                break

    score = (
        matches
        / possible
    )

    return clamp(
        score
    )


# =============================================================================
# 11. OBJECT / CONCEPT COMPATIBILITY
# =============================================================================

def calculate_object_compatibility(
    semantic_profile,
    candidate
):

    """
    Prototype concept compatibility.

    This replaces the misleading claim of object detection.

    The score measures compatibility between abstract visual concepts
    and candidate semantic tags.
    """

    concepts = (
        semantic_profile[
            "semantic_concepts"
        ]
    )

    candidate_tags = (
        candidate[
            "scene_tags"
        ]
        +
        candidate[
            "environment_tags"
        ]
    )

    if not concepts:

        return 0.50

    matches = 0

    for concept in concepts:

        for tag in candidate_tags:

            if (
                tag in concept
                or concept in tag
            ):

                matches += 1

                break

    score = (
        matches
        / len(concepts)
    )

    return clamp(
        score
    )


# =============================================================================
# 12. ENVIRONMENT COMPATIBILITY
# =============================================================================

def calculate_environment_compatibility(
    semantic_profile,
    candidate
):

    environment_label = (
        semantic_profile[
            "environment"
        ][
            "label"
        ].lower()
    )

    environment_confidence = (
        semantic_profile[
            "environment"
        ][
            "confidence"
        ]
    )

    score = 0.30

    for tag in candidate[
        "environment_tags"
    ]:

        if (
            tag in environment_label
        ):

            score += 0.40

        if (
            "open"
            in environment_label
            and tag == "outdoor"
        ):

            score += 0.20

        if (
            "dark"
            in environment_label
            and tag == "enclosed"
        ):

            score += 0.20

        if (
            "mixed"
            in environment_label
            and tag == "mixed"
        ):

            score += 0.25

    score = (
        score
        * environment_confidence
    )

    return clamp(
        score
    )


# =============================================================================
# 13. MOOD COMPATIBILITY
# =============================================================================

def calculate_mood_compatibility(
    semantic_profile,
    candidate
):

    image_mood = (
        semantic_profile[
            "mood"
        ]
    )

    candidate_mood = (
        candidate[
            "mood_profile"
        ]
    )

    valence_distance = abs(

        image_mood[
            "valence"
        ]

        -

        candidate_mood[
            "valence"
        ]
    )

    arousal_distance = abs(

        image_mood[
            "arousal"
        ]

        -

        candidate_mood[
            "arousal"
        ]
    )

    average_distance = (

        valence_distance
        + arousal_distance

    ) / 2

    similarity = (
        1.0
        - average_distance
    )

    return clamp(
        similarity
    )


# =============================================================================
# 14. SEMANTIC SIMILARITY
# =============================================================================

def calculate_semantic_similarity(
    semantic_profile,
    candidate
):

    """
    Aggregate semantic similarity.

    This prototype combines:

    - Preliminary candidate relevance
    - Scene confidence
    - Mood compatibility
    """

    preliminary = (
        candidate[
            "preliminary_score"
        ]
    )

    scene_confidence = (
        semantic_profile[
            "scene"
        ][
            "confidence"
        ]
    )

    mood_score = (
        calculate_mood_compatibility(
            semantic_profile,
            candidate
        )
    )

    score = (

        0.45
        * preliminary

        +

        0.25
        * scene_confidence

        +

        0.30
        * mood_score
    )

    return clamp(
        score
    )


# =============================================================================
# 15. EXPLICIT CONFLICT DETECTION
# =============================================================================

CONFLICT_RULES = {

    "natural": {

        "conflicts":
            [
                "traffic",
                "car engine",
                "heavy concert"
            ]
    },

    "calm": {

        "conflicts":
            [
                "heavy concert",
                "audience applause"
            ]
    },

    "low-light": {

        "conflicts":
            [
                "bird chirps"
            ]
    }
}


def detect_conflicts(
    semantic_profile,
    candidate
):

    """
    Detect explicit semantic contradictions.

    Returns:

    penalty
    conflict explanations
    """

    candidate_name = (
        candidate[
            "concept"
        ].lower()
    )

    profile_text = " ".join(

        [

            semantic_profile[
                "scene"
            ][
                "label"
            ].lower(),

            semantic_profile[
                "environment"
            ][
                "label"
            ].lower(),

            semantic_profile[
                "mood"
            ][
                "label"
            ].lower(),

            " ".join(
                semantic_profile[
                    "semantic_concepts"
                ]
            ).lower()
        ]
    )

    penalty = 0.0

    conflicts_found = []

    # -------------------------------------------------------------------------
    # NATURAL / URBAN CONFLICT
    # -------------------------------------------------------------------------

    natural_signals = [

        "natural",
        "cool-toned",
        "possible nature"
    ]

    if any(
        signal in profile_text
        for signal in natural_signals
    ):

        urban_conflicts = [

            "traffic",
            "car engine"
        ]

        for conflict in urban_conflicts:

            if conflict in candidate_name:

                penalty += 0.30

                conflicts_found.append(

                    f"Natural/open visual hypothesis "
                    f"conflicts with '{candidate['concept']}'."
                )

    # -------------------------------------------------------------------------
    # CALM / HIGH-INTENSITY CONFLICT
    # -------------------------------------------------------------------------

    if (
        semantic_profile[
            "mood"
        ][
            "arousal"
        ]
        < 0.45
    ):

        intense_conflicts = [

            "heavy concert",
            "audience applause"
        ]

        for conflict in intense_conflicts:

            if conflict in candidate_name:

                penalty += 0.25

                conflicts_found.append(

                    f"Low-arousal visual mood conflicts "
                    f"with the high-intensity candidate "
                    f"'{candidate['concept']}'."
                )

    # -------------------------------------------------------------------------
    # LOW-LIGHT / BIRD CONFLICT
    # -------------------------------------------------------------------------

    if (
        "low-light"
        in profile_text
    ):

        if (
            "bird chirps"
            in candidate_name
        ):

            penalty += 0.15

            conflicts_found.append(

                "Low-light scene hypothesis weakly "
                "conflicts with daytime bird activity."
            )

    penalty = clamp(
        penalty,
        0.0,
        0.70
    )

    return penalty, conflicts_found


# =============================================================================
# 16. SCORE STABILITY
# =============================================================================

def calculate_score_stability(
    scene_score,
    object_score,
    environment_score,
    mood_score,
    semantic_score
):

    """
    A resilient decision should not depend entirely on one signal.

    Stability measures agreement among evidence dimensions.
    """

    scores = np.array([

        scene_score,

        object_score,

        environment_score,

        mood_score,

        semantic_score
    ])

    variance = float(
        np.var(scores)
    )

    stability = (
        1.0
        - min(
            variance
            * 4.0,
            1.0
        )
    )

    return clamp(
        stability
    )


# =============================================================================
# 17. DECISION CONFIDENCE
# =============================================================================

def calculate_decision_confidence(
    semantic_confidence,
    score_stability,
    conflict_penalty,
    final_score
):

    confidence = (

        0.30
        * semantic_confidence

        +

        0.30
        * score_stability

        +

        0.20
        * (
            1.0
            - conflict_penalty
        )

        +

        0.20
        * max(
            final_score,
            1.0
            - final_score
        )
    )

    return clamp(
        confidence
    )


# =============================================================================
# 18. REPAIR SUGGESTION ENGINE
# =============================================================================

def generate_repair_suggestion(
    semantic_profile,
    candidate,
    verification_scores
):

    """
    Generate a lightweight repair recommendation.

    The purpose is not to synthesize a full semantic prompt,
    but to transform a partially compatible candidate into
    a more context-aware concept.
    """

    scene = (
        semantic_profile[
            "scene"
        ][
            "label"
        ]
    )

    mood = (
        semantic_profile[
            "mood"
        ][
            "label"
        ]
    )

    candidate_name = (
        candidate[
            "concept"
        ]
    )

    weak_dimensions = []

    if (
        verification_scores[
            "scene_score"
        ]
        < 0.50
    ):

        weak_dimensions.append(
            "scene alignment"
        )

    if (
        verification_scores[
            "environment_score"
        ]
        < 0.50
    ):

        weak_dimensions.append(
            "environment alignment"
        )

    if (
        verification_scores[
            "mood_score"
        ]
        < 0.55
    ):

        weak_dimensions.append(
            "mood alignment"
        )

    if weak_dimensions:

        weakness_text = (
            ", ".join(
                weak_dimensions
            )
        )

    else:

        weakness_text = (
            "overall semantic specificity"
        )

    suggestion = (

        f"Repair '{candidate_name}' by adding "
        f"context consistent with the inferred "
        f"scene '{scene}' and mood '{mood}'. "
        f"Primary improvement target: {weakness_text}."
    )

    return suggestion


# =============================================================================
# 19. CAS-V 2.0 MULTI-DIMENSIONAL VERIFIER
# =============================================================================

def casv_multi_stage_verifier(
    semantic_profile,
    candidates,
    accept_threshold=0.75,
    repair_threshold=0.45,
    uncertainty_threshold=0.40
):

    """
    CAS-V 2.0 verification.

    Four possible decisions:

    ACCEPT
    REPAIR
    REJECT
    UNCERTAIN
    """

    results = []

    accepted = []

    repaired = []

    rejected = []

    uncertain = []

    semantic_confidence = (
        semantic_profile[
            "semantic_confidence"
        ]
    )

    for candidate in candidates:

        # ---------------------------------------------------------------------
        # DIMENSION 1
        # ---------------------------------------------------------------------

        scene_score = (
            calculate_scene_compatibility(
                semantic_profile,
                candidate
            )
        )

        # ---------------------------------------------------------------------
        # DIMENSION 2
        # ---------------------------------------------------------------------

        object_score = (
            calculate_object_compatibility(
                semantic_profile,
                candidate
            )
        )

        # ---------------------------------------------------------------------
        # DIMENSION 3
        # ---------------------------------------------------------------------

        environment_score = (
            calculate_environment_compatibility(
                semantic_profile,
                candidate
            )
        )

        # ---------------------------------------------------------------------
        # DIMENSION 4
        # ---------------------------------------------------------------------

        mood_score = (
            calculate_mood_compatibility(
                semantic_profile,
                candidate
            )
        )

        # ---------------------------------------------------------------------
        # DIMENSION 5
        # ---------------------------------------------------------------------

        semantic_score = (
            calculate_semantic_similarity(
                semantic_profile,
                candidate
            )
        )

        # ---------------------------------------------------------------------
        # CONFLICT DETECTION
        # ---------------------------------------------------------------------

        conflict_penalty, conflicts = (

            detect_conflicts(
                semantic_profile,
                candidate
            )
        )

        # ---------------------------------------------------------------------
        # WEIGHTED AGGREGATION
        # ---------------------------------------------------------------------

        base_score = (

            0.22
            * scene_score

            +

            0.18
            * object_score

            +

            0.16
            * environment_score

            +

            0.18
            * mood_score

            +

            0.26
            * semantic_score
        )

        final_score = clamp(

            base_score
            - conflict_penalty
        )

        # ---------------------------------------------------------------------
        # STABILITY
        # ---------------------------------------------------------------------

        stability = (

            calculate_score_stability(

                scene_score,

                object_score,

                environment_score,

                mood_score,

                semantic_score
            )
        )

        # ---------------------------------------------------------------------
        # DECISION CONFIDENCE
        # ---------------------------------------------------------------------

        decision_confidence = (

            calculate_decision_confidence(

                semantic_confidence,

                stability,

                conflict_penalty,

                final_score
            )
        )

        # ---------------------------------------------------------------------
        # DECISION
        # ---------------------------------------------------------------------

        decision = None

        reason = ""

        repair_suggestion = None

        # If visual semantic confidence is too low,
        # the system should avoid overconfident verification.

        if (
            semantic_confidence
            < uncertainty_threshold
        ):

            decision = "UNCERTAIN"

            reason = (

                "The visual semantic profile has insufficient "
                "confidence for a reliable verification decision."
            )

        elif (
            decision_confidence
            < uncertainty_threshold
        ):

            decision = "UNCERTAIN"

            reason = (

                "Evidence dimensions show insufficient agreement "
                "or decision confidence."
            )

        elif (
            final_score
            >= accept_threshold
        ):

            decision = "ACCEPT"

            reason = (

                "The candidate demonstrates strong multi-dimensional "
                "compatibility with the current semantic profile."
            )

        elif (
            final_score
            >= repair_threshold
        ):

            decision = "REPAIR"

            reason = (

                "The candidate is partially compatible but contains "
                "weak semantic alignment that can potentially be improved."
            )

        else:

            decision = "REJECT"

            reason = (

                "The candidate does not demonstrate sufficient "
                "semantic compatibility with the current profile."
            )

        # ---------------------------------------------------------------------
        # ADD CONFLICT INFORMATION
        # ---------------------------------------------------------------------

        if conflicts:

            reason += (

                " Explicit conflicts were detected: "

                + " ".join(conflicts)
            )

        # ---------------------------------------------------------------------
        # REPAIR
        # ---------------------------------------------------------------------

        verification_scores = {

            "scene_score":
                scene_score,

            "object_score":
                object_score,

            "environment_score":
                environment_score,

            "mood_score":
                mood_score,

            "semantic_score":
                semantic_score
        }

        if decision == "REPAIR":

            repair_suggestion = (

                generate_repair_suggestion(

                    semantic_profile,

                    candidate,

                    verification_scores
                )
            )

        result = {

            "concept":
                candidate[
                    "concept"
                ],

            "category":
                candidate[
                    "category"
                ],

            "preliminary_score":
                candidate[
                    "preliminary_score"
                ],

            "scene_score":
                scene_score,

            "object_score":
                object_score,

            "environment_score":
                environment_score,

            "mood_score":
                mood_score,

            "semantic_score":
                semantic_score,

            "conflict_penalty":
                conflict_penalty,

            "conflicts":
                conflicts,

            "score_stability":
                stability,

            "final_score":
                final_score,

            "decision_confidence":
                decision_confidence,

            "decision":
                decision,

            "reason":
                reason,

            "repair_suggestion":
                repair_suggestion
        }

        results.append(
            result
        )

        if decision == "ACCEPT":

            accepted.append(
                result
            )

        elif decision == "REPAIR":

            repaired.append(
                result
            )

        elif decision == "REJECT":

            rejected.append(
                result
            )

        else:

            uncertain.append(
                result
            )

    results = sorted(

        results,

        key=lambda item:
            item[
                "final_score"
            ],

        reverse=True
    )

    return {

        "semantic_confidence":
            semantic_confidence,

        "results":
            results,

        "accepted":
            accepted,

        "repair":
            repaired,

        "rejected":
            rejected,

        "uncertain":
            uncertain
    }


# =============================================================================
# 20. AUDIO SYNTHESIS
# =============================================================================

CONCEPT_FREQUENCIES = {

    "Ocean Waves":
        90,

    "Bird Chirps":
        880,

    "Forest Ambience":
        150,

    "Piano Music":
        220,

    "Soft Piano":
        196,

    "Ambient Synth":
        260,

    "Car Engine":
        110,

    "Traffic Ambience":
        140,

    "Audience Applause":
        330,

    "Heavy Concert":
        70
}


def concept_to_frequency(
    concept
):

    return (
        CONCEPT_FREQUENCIES.get(
            concept,
            220
        )
    )


def generate_verified_audio(
    accepted_concepts,
    repaired_concepts,
    output_path,
    sample_rate=22050,
    duration=4.0
):

    """
    Lightweight synthetic audio demonstration.

    ACCEPTED concepts receive full contribution.

    REPAIRED concepts receive reduced contribution.

    This demonstrates how verification decisions influence
    the output generation stage.
    """

    total_samples = int(
        sample_rate
        * duration
    )

    time_axis = np.linspace(

        0,

        duration,

        total_samples,

        endpoint=False
    )

    waveform = np.zeros_like(
        time_axis
    )

    # -------------------------------------------------------------------------
    # ACCEPTED CONCEPTS
    # -------------------------------------------------------------------------

    for item in accepted_concepts:

        frequency = (
            concept_to_frequency(
                item["concept"]
            )
        )

        amplitude = (

            0.12

            +

            item[
                "final_score"
            ]

            * 0.28
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

    # -------------------------------------------------------------------------
    # REPAIRED CONCEPTS
    # -------------------------------------------------------------------------

    for item in repaired_concepts:

        frequency = (
            concept_to_frequency(
                item["concept"]
            )
        )

        amplitude = (

            0.05

            +

            item[
                "final_score"
            ]

            * 0.12
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

    # -------------------------------------------------------------------------
    # EMPTY FALLBACK
    # -------------------------------------------------------------------------

    if np.max(
        np.abs(waveform)
    ) == 0:

        waveform = (

            0.05

            * np.sin(

                2

                * np.pi

                * 220

                * time_axis
            )
        )

    # -------------------------------------------------------------------------
    # NORMALIZATION
    # -------------------------------------------------------------------------

    waveform = (

        waveform

        / (

            np.max(
                np.abs(waveform)
            )

            + 1e-8
        )
    )

    waveform = (

        waveform
        * 0.8
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


# =============================================================================
# 21. EXPERIMENT STORAGE
# =============================================================================

def save_experiment(
    image_path,
    analysis,
    semantic_profile,
    candidates,
    verification,
    audio_path=None
):

    timestamp = datetime.now().strftime(
        "%Y%m%d_%H%M%S"
    )

    experiment_id = (

        f"{timestamp}_"

        f"{uuid.uuid4().hex[:6]}"
    )

    experiment_path = (

        EXPERIMENT_DIR

        / experiment_id
    )

    experiment_path.mkdir(
        exist_ok=True
    )

    # -------------------------------------------------------------------------
    # IMAGE
    # -------------------------------------------------------------------------

    shutil.copy2(

        image_path,

        experiment_path
        / "input_image"
        + Path(image_path).suffix
    )


# =============================================================================
# FIXED SAVE FUNCTION
# =============================================================================

def save_complete_experiment(
    image_path,
    analysis,
    semantic_profile,
    candidates,
    verification,
    audio_path=None
):

    timestamp = datetime.now().strftime(
        "%Y%m%d_%H%M%S"
    )

    experiment_id = (

        f"{timestamp}_"

        f"{uuid.uuid4().hex[:6]}"
    )

    experiment_path = (
        EXPERIMENT_DIR
        / experiment_id
    )

    experiment_path.mkdir(
        exist_ok=True
    )

    # -------------------------------------------------------------------------
    # SAVE IMAGE
    # -------------------------------------------------------------------------

    image = Image.open(
        image_path
    )

    image.save(
        experiment_path
        / "input_image.png"
    )

    # -------------------------------------------------------------------------
    # SAVE ANALYSIS
    # -------------------------------------------------------------------------

    with open(

        experiment_path
        / "visual_analysis.json",

        "w",

        encoding="utf-8"

    ) as file:

        json.dump(

            analysis,

            file,

            indent=4
        )

    # -------------------------------------------------------------------------
    # SAVE SEMANTIC PROFILE
    # -------------------------------------------------------------------------

    with open(

        experiment_path
        / "semantic_profile.json",

        "w",

        encoding="utf-8"

    ) as file:

        json.dump(

            semantic_profile,

            file,

            indent=4
        )

    # -------------------------------------------------------------------------
    # SAVE CANDIDATES
    # -------------------------------------------------------------------------

    with open(

        experiment_path
        / "candidates.json",

        "w",

        encoding="utf-8"

    ) as file:

        json.dump(

            candidates,

            file,

            indent=4
        )

    # -------------------------------------------------------------------------
    # SAVE VERIFICATION
    # -------------------------------------------------------------------------

    with open(

        experiment_path
        / "casv_verification.json",

        "w",

        encoding="utf-8"

    ) as file:

        json.dump(

            verification,

            file,

            indent=4
        )

    # -------------------------------------------------------------------------
    # SAVE AUDIO
    # -------------------------------------------------------------------------

    if (

        audio_path

        and

        os.path.exists(
            audio_path
        )
    ):

        shutil.copy2(

            audio_path,

            experiment_path
            / "verified_audio.wav"
        )

    return experiment_path


# =============================================================================
# 22. LOAD EXPERIMENTS
# =============================================================================

def get_experiments():

    experiments = [

        item

        for item in EXPERIMENT_DIR.iterdir()

        if item.is_dir()
    ]

    return sorted(

        experiments,

        reverse=True
    )


def load_json_file(
    file_path
):

    if not file_path.exists():

        return None

    try:

        with open(

            file_path,

            "r",

            encoding="utf-8"

        ) as file:

            return json.load(
                file
            )

    except Exception:

        return None


# =============================================================================
# 23. SCORE BAR COMPONENT
# =============================================================================

def display_score_bar(
    label,
    score
):

    percentage = (
        clamp(score)
        * 100
    )

    st.markdown(

        f"""
<div class="score-label">

{label}: {percentage:.1f}%

</div>

<div class="score-container">

<div
class="score-fill"
style="width:{percentage:.1f}%">
</div>

</div>
""",

        unsafe_allow_html=True
    )


# =============================================================================
# 24. APP HEADER
# =============================================================================

st.markdown(
    """
<div class="casv-hero">

<div class="casv-badge">
CAS-V 2.0 • MULTI-STAGE RESEARCH PROTOTYPE
</div>

<div class="casv-title">
Conflict-Aware Semantic Verification
</div>

<div class="casv-subtitle">

A resilient multi-dimensional semantic verification framework for
image-to-audio generation. CAS-V 2.0 combines visual evidence,
semantic compatibility, mood alignment, explicit conflict detection,
score stability, uncertainty estimation, and repair decisions.

</div>

</div>
""",
    unsafe_allow_html=True
)


# =============================================================================
# 25. SIDEBAR
# =============================================================================

st.sidebar.markdown(
    "# 🛡️ CAS-V 2.0"
)

st.sidebar.caption(
    "Explainable Semantic Verification"
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

st.sidebar.markdown(
    "---"
)

st.sidebar.caption(
    """
Decision states:

✓ ACCEPT

🔧 REPAIR

✗ REJECT

? UNCERTAIN
"""
)


# =============================================================================
# PAGE 1 — RESEARCH OVERVIEW
# =============================================================================

if page == "🏠 Research Overview":

    st.markdown(
        "## CAS-V 2.0 Research Architecture"
    )

    st.markdown(
        """
<div class="research-card">

<h4>Research Objective</h4>

CAS-V introduces a verification layer between visual interpretation
and final audio synthesis. Rather than relying on a single
accept/reject rule, the system aggregates multiple forms of evidence,
detects explicit semantic conflicts, estimates decision stability,
and avoids overconfident decisions when the visual evidence is weak.

</div>
""",
        unsafe_allow_html=True
    )

    st.markdown(
        "### Multi-Stage Pipeline"
    )

    columns = st.columns(
        6
    )

    steps = [

        (
            "🖼️",
            "Image",
            "Visual input"
        ),

        (
            "👁️",
            "Analysis",
            "Multi-signal visual evidence"
        ),

        (
            "🧠",
            "Profile",
            "Structured semantic profile"
        ),

        (
            "🎼",
            "Candidates",
            "Neutral candidate generation"
        ),

        (
            "🛡️",
            "CAS-V",
            "Multi-dimensional verification"
        ),

        (
            "🔊",
            "Output",
            "Verified prototype audio"
        )
    ]

    for column, step in zip(
        columns,
        steps
    ):

        icon, title, description = step

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

    st.markdown(
        "---"
    )

    st.markdown(
        "### Verification Evidence"
    )

    evidence_columns = st.columns(
        3
    )

    evidence = [

        (
            "🌄 Scene Compatibility",
            "Does the candidate align with the inferred visual scene?"
        ),

        (
            "🧩 Concept Compatibility",
            "Does the candidate align with abstract semantic concepts?"
        ),

        (
            "🌍 Environment Compatibility",
            "Does the candidate fit the inferred environment?"
        ),

        (
            "😊 Mood Compatibility",
            "Does the candidate align with estimated valence and arousal?"
        ),

        (
            "🔗 Semantic Similarity",
            "Does aggregated semantic evidence support the candidate?"
        ),

        (
            "⚠️ Conflict Detection",
            "Does explicit contradictory evidence exist?"
        )
    ]

    for column, item in zip(
        evidence_columns * 2,
        evidence
    ):

        title, description = item

        with column:

            st.markdown(

                f"""
<div class="research-card">

<h4>{title}</h4>

<p>{description}</p>

</div>
""",

                unsafe_allow_html=True
            )

    st.markdown(
        "---"
    )

    st.markdown(
        "### Resilience Layer"
    )

    resilience_left, resilience_right = st.columns(
        2
    )

    with resilience_left:

        st.markdown(
            """
<div class="research-card">

<h4>Uncertainty Awareness</h4>

When the visual semantic profile has insufficient confidence,
CAS-V can return <b>UNCERTAIN</b> rather than making a
potentially unreliable accept/reject decision.

</div>
""",
            unsafe_allow_html=True
        )

    with resilience_right:

        st.markdown(
            """
<div class="research-card">

<h4>Evidence Stability</h4>

CAS-V measures agreement across scene, concept, environment,
mood, and semantic dimensions. A decision supported by highly
inconsistent evidence receives lower confidence.

</div>
""",
            unsafe_allow_html=True
        )

    experiments = get_experiments()

    st.markdown(
        "---"
    )

    st.markdown(
        "### Experiment Summary"
    )

    total_experiments = len(
        experiments
    )

    accepted_total = 0

    repaired_total = 0

    rejected_total = 0

    uncertain_total = 0

    for experiment in experiments:

        verification = load_json_file(

            experiment
            / "casv_verification.json"
        )

        if verification:

            accepted_total += len(
                verification.get(
                    "accepted",
                    []
                )
            )

            repaired_total += len(
                verification.get(
                    "repair",
                    []
                )
            )

            rejected_total += len(
                verification.get(
                    "rejected",
                    []
                )
            )

            uncertain_total += len(
                verification.get(
                    "uncertain",
                    []
                )
            )

    metric_columns = st.columns(
        5
    )

    metric_columns[0].metric(
        "Experiments",
        total_experiments
    )

    metric_columns[1].metric(
        "Accepted",
        accepted_total
    )

    metric_columns[2].metric(
        "Repair",
        repaired_total
    )

    metric_columns[3].metric(
        "Rejected",
        rejected_total
    )

    metric_columns[4].metric(
        "Uncertain",
        uncertain_total
    )


# =============================================================================
# PAGE 2 — CAS-V DEMO
# =============================================================================

elif page == "🎵 CAS-V Demo":

    st.markdown(
        "## Interactive CAS-V 2.0 Demonstration"
    )

    st.caption(
        """
Run the complete multi-stage verification pipeline:
image → semantic profile → candidate pool → evidence matrix →
resilient decision → verified audio.
"""
    )

    # -------------------------------------------------------------------------
    # INPUT
    # -------------------------------------------------------------------------

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

        key="casv_upload"
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

        image_column, information_column = st.columns(
            [1.1, 1],
            gap="large"
        )

        with image_column:

            st.image(

                Image.open(
                    image_path
                ),

                use_container_width=True,

                caption="Research Input Image"
            )

        with information_column:

            st.markdown(
                """
<div class="info-card">

<b>Prototype Transparency</b>

<br><br>

CAS-V 2.0 currently uses multi-signal visual statistics to build
a heuristic semantic profile. The prototype does not claim
object-level recognition unless a dedicated vision model is added.

</div>
""",
                unsafe_allow_html=True
            )

            run_button = st.button(

                "▶ Run CAS-V 2.0 Verification",

                use_container_width=True
            )

        # ---------------------------------------------------------------------
        # RUN PIPELINE
        # ---------------------------------------------------------------------

        if run_button:

            # =================================================================
            # STEP 2 — VISUAL ANALYSIS
            # =================================================================

            analysis = analyze_uploaded_image(
                image_path
            )

            semantic_profile = build_semantic_profile(
                analysis
            )

            st.session_state[
                "last_analysis"
            ] = analysis

            st.session_state[
                "last_semantic_profile"
            ] = semantic_profile

            st.markdown(
                "---"
            )

            st.markdown(
                "### ② Multi-Signal Visual Analysis"
            )

            feature_columns = st.columns(
                5
            )

            visual_features = (
                analysis[
                    "visual_features"
                ]
            )

            feature_columns[0].metric(
                "Brightness",
                f"{visual_features['brightness']:.2f}"
            )

            feature_columns[1].metric(
                "Contrast",
                f"{visual_features['contrast']:.2f}"
            )

            feature_columns[2].metric(
                "Complexity",
                f"{visual_features['complexity']:.2f}"
            )

            feature_columns[3].metric(
                "Dominant Color",
                visual_features[
                    "dominant_color"
                ].title()
            )

            feature_columns[4].metric(
                "Semantic Confidence",
                f"{analysis['semantic_confidence']:.2f}"
            )

            with st.expander(
                "View detailed visual signals"
            ):

                st.json(
                    analysis
                )

            # =================================================================
            # STEP 3 — SEMANTIC PROFILE
            # =================================================================

            st.markdown(
                "---"
            )

            st.markdown(
                "### ③ Structured Semantic Profile"
            )

            profile_left, profile_right = st.columns(
                2,
                gap="large"
            )

            with profile_left:

                st.markdown(
                    """
<div class="research-card">
""",
                    unsafe_allow_html=True
                )

                st.markdown(
                    "#### Scene and Environment"
                )

                st.write(

                    f"**Scene Hypothesis:** "
                    f"{semantic_profile['scene']['label']}"
                )

                st.write(

                    f"**Scene Confidence:** "
                    f"{semantic_profile['scene']['confidence']:.2f}"
                )

                st.write(

                    f"**Environment:** "
                    f"{semantic_profile['environment']['label']}"
                )

                st.write(

                    f"**Environment Confidence:** "
                    f"{semantic_profile['environment']['confidence']:.2f}"
                )

                st.markdown(
                    "</div>",
                    unsafe_allow_html=True
                )

            with profile_right:

                st.markdown(
                    """
<div class="research-card">
""",
                    unsafe_allow_html=True
                )

                st.markdown(
                    "#### Mood and Semantic Concepts"
                )

                st.write(

                    f"**Mood:** "
                    f"{semantic_profile['mood']['label']}"
                )

                st.write(

                    f"**Valence:** "
                    f"{semantic_profile['mood']['valence']:.2f}"
                )

                st.write(

                    f"**Arousal:** "
                    f"{semantic_profile['mood']['arousal']:.2f}"
                )

                concepts = (
                    semantic_profile[
                        "semantic_concepts"
                    ]
                )

                if concepts:

                    st.write(
                        "**Abstract Concepts:** "
                        + ", ".join(concepts)
                    )

                else:

                    st.write(
                        "**Abstract Concepts:** None"
                    )

                st.markdown(
                    "</div>",
                    unsafe_allow_html=True
                )

            # =================================================================
            # STEP 4 — CANDIDATES
            # =================================================================

            st.markdown(
                "---"
            )

            st.markdown(
                "### ④ Neutral Audio Candidate Pool"
            )

            candidates = generate_audio_candidates(
                semantic_profile
            )

            st.session_state[
                "last_candidates"
            ] = candidates

            candidate_rows = []

            for item in candidates:

                candidate_rows.append({

                    "Concept":
                        item["concept"],

                    "Category":
                        item["category"],

                    "Preliminary Relevance":
                        f"{item['preliminary_score'] * 100:.1f}%"
                })

            st.dataframe(

                pd.DataFrame(
                    candidate_rows
                ),

                hide_index=True,

                use_container_width=True
            )

            st.caption(
                """
These preliminary scores are candidate-generation relevance estimates.
They are not CAS-V decisions.
"""
            )

            # =================================================================
            # STEP 5 — CAS-V VERIFICATION
            # =================================================================

            st.markdown(
                "---"
            )

            st.markdown(
                "### ⑤ Multi-Dimensional CAS-V Verification"
            )

            verification = casv_multi_stage_verifier(

                semantic_profile,

                candidates,

                accept_threshold=
                    st.session_state[
                        "accept_threshold"
                    ],

                repair_threshold=
                    st.session_state[
                        "repair_threshold"
                    ],

                uncertainty_threshold=
                    st.session_state[
                        "uncertainty_threshold"
                    ]
            )

            st.session_state[
                "last_verification"
            ] = verification

            total = len(
                verification[
                    "results"
                ]
            )

            accepted = verification[
                "accepted"
            ]

            repaired = verification[
                "repair"
            ]

            rejected = verification[
                "rejected"
            ]

            uncertain = verification[
                "uncertain"
            ]

            st.markdown(

                f"""
<div class="verification-summary">

<div class="verification-title">

🛡️ CAS-V 2.0 Decision Summary

</div>

<b>Semantic Profile Confidence:</b>
{verification['semantic_confidence']:.2f}

<br><br>

<b>Total Candidates:</b>
{total}

&nbsp;&nbsp; • &nbsp;&nbsp;

<b>Accepted:</b>
{len(accepted)}

&nbsp;&nbsp; • &nbsp;&nbsp;

<b>Repair:</b>
{len(repaired)}

&nbsp;&nbsp; • &nbsp;&nbsp;

<b>Rejected:</b>
{len(rejected)}

&nbsp;&nbsp; • &nbsp;&nbsp;

<b>Uncertain:</b>
{len(uncertain)}

</div>
""",

                unsafe_allow_html=True
            )

            # =================================================================
            # VERIFICATION MATRIX
            # =================================================================

            matrix_rows = []

            for result in verification[
                "results"
            ]:

                matrix_rows.append({

                    "Candidate":
                        result["concept"],

                    "Scene":
                        f"{result['scene_score']:.2f}",

                    "Concept":
                        f"{result['object_score']:.2f}",

                    "Environment":
                        f"{result['environment_score']:.2f}",

                    "Mood":
                        f"{result['mood_score']:.2f}",

                    "Semantic":
                        f"{result['semantic_score']:.2f}",

                    "Conflict":
                        f"{result['conflict_penalty']:.2f}",

                    "Stability":
                        f"{result['score_stability']:.2f}",

                    "Final":
                        f"{result['final_score']:.2f}",

                    "Decision":
                        result["decision"]
                })

            st.markdown(
                "#### CAS-V Evidence Matrix"
            )

            st.dataframe(

                pd.DataFrame(
                    matrix_rows
                ),

                hide_index=True,

                use_container_width=True
            )

            # =================================================================
            # EVIDENCE EXPLORER
            # =================================================================

            st.markdown(
                "#### Evidence Explorer"
            )

            for result in verification[
                "results"
            ]:

                decision_icon = {

                    "ACCEPT": "✓",

                    "REPAIR": "🔧",

                    "REJECT": "✗",

                    "UNCERTAIN": "?"
                }.get(
                    result["decision"],
                    "•"
                )

                with st.expander(

                    f"{decision_icon} "
                    f"{result['concept']} "
                    f"— {result['decision']} "
                    f"({result['final_score']:.2f})"
                ):

                    evidence_left, evidence_right = st.columns(
                        2
                    )

                    with evidence_left:

                        display_score_bar(
                            "Scene Compatibility",
                            result[
                                "scene_score"
                            ]
                        )

                        display_score_bar(
                            "Concept Compatibility",
                            result[
                                "object_score"
                            ]
                        )

                        display_score_bar(
                            "Environment Compatibility",
                            result[
                                "environment_score"
                            ]
                        )

                    with evidence_right:

                        display_score_bar(
                            "Mood Compatibility",
                            result[
                                "mood_score"
                            ]
                        )

                        display_score_bar(
                            "Semantic Similarity",
                            result[
                                "semantic_score"
                            ]
                        )

                        display_score_bar(
                            "Evidence Stability",
                            result[
                                "score_stability"
                            ]
                        )

                    st.metric(

                        "Final Compatibility Score",

                        f"{result['final_score']:.3f}"
                    )

                    st.metric(

                        "Decision Confidence",

                        f"{result['decision_confidence']:.3f}"
                    )

                    st.write(
                        "**Decision Explanation**"
                    )

                    st.write(
                        result["reason"]
                    )

                    if result[
                        "conflicts"
                    ]:

                        st.warning(
                            "Detected Conflicts"
                        )

                        for conflict in result[
                            "conflicts"
                        ]:

                            st.write(
                                f"• {conflict}"
                            )

                    if (
                        result[
                            "repair_suggestion"
                        ]
                    ):

                        st.info(
                            result[
                                "repair_suggestion"
                            ]
                        )

            # =================================================================
            # DECISION CATEGORIES
            # =================================================================

            st.markdown(
                "---"
            )

            st.markdown(
                "### ⑥ Decision Categories"
            )

            decision_columns = st.columns(
                4
            )

            with decision_columns[0]:

                st.markdown(
                    "#### ✓ ACCEPT"
                )

                for item in accepted:

                    st.markdown(

                        f"""
<div class="accept-card">

<b>{item['concept']}</b>

<br><br>

Final Score:
{item['final_score']:.2f}

<br>

Confidence:
{item['decision_confidence']:.2f}

</div>
""",

                        unsafe_allow_html=True
                    )

            with decision_columns[1]:

                st.markdown(
                    "#### 🔧 REPAIR"
                )

                for item in repaired:

                    st.markdown(

                        f"""
<div class="repair-card">

<b>{item['concept']}</b>

<br><br>

Final Score:
{item['final_score']:.2f}

<br><br>

{item['repair_suggestion']}

</div>
""",

                        unsafe_allow_html=True
                    )

            with decision_columns[2]:

                st.markdown(
                    "#### ✗ REJECT"
                )

                for item in rejected:

                    st.markdown(

                        f"""
<div class="reject-card">

<b>{item['concept']}</b>

<br><br>

Final Score:
{item['final_score']:.2f}

</div>
""",

                        unsafe_allow_html=True
                    )

            with decision_columns[3]:

                st.markdown(
                    "#### ? UNCERTAIN"
                )

                for item in uncertain:

                    st.markdown(

                        f"""
<div class="uncertain-card">

<b>{item['concept']}</b>

<br><br>

{item['reason']}

</div>
""",

                        unsafe_allow_html=True
                    )

            # =================================================================
            # STEP 7 — AUDIO
            # =================================================================

            st.markdown(
                "---"
            )

            st.markdown(
                "### ⑦ Verified / Repaired Audio Output"
            )

            if accepted or repaired:

                temporary_audio_path = (

                    TEMP_DIR

                    / "current_verified_audio.wav"
                )

                generate_verified_audio(

                    accepted,

                    repaired,

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

                    f"Audio generated from "
                    f"{len(accepted)} accepted and "
                    f"{len(repaired)} repaired concept(s)."
                )

                st.audio(

                    str(
                        temporary_audio_path
                    ),

                    format="audio/wav"
                )

                audio_columns = st.columns(
                    2
                )

                with audio_columns[0]:

                    with open(

                        temporary_audio_path,

                        "rb"

                    ) as audio_file:

                        st.download_button(

                            "📥 Download CAS-V Audio",

                            audio_file.read(),

                            file_name=
                                "casv2_verified_audio.wav",

                            mime=
                                "audio/wav",

                            use_container_width=True
                        )

                with audio_columns[1]:

                    if st.button(

                        "💾 Save Complete Experiment",

                        use_container_width=True
                    ):

                        experiment_path = (

                            save_complete_experiment(

                                image_path,

                                analysis,

                                semantic_profile,

                                candidates,

                                verification,

                                temporary_audio_path
                            )
                        )

                        st.session_state[
                            "last_experiment_id"
                        ] = (
                            experiment_path.name
                        )

                        st.success(

                            f"Experiment saved: "
                            f"{experiment_path.name}"
                        )

            else:

                st.warning(

                    """
No candidate reached ACCEPT or REPAIR status.

The system therefore avoided audio generation because the
available evidence did not support a sufficiently reliable
semantic output.
"""
                )

    else:

        st.markdown(
            """
<div class="research-card">

<h4>🖼️ Start a CAS-V 2.0 Experiment</h4>

Upload an image to execute the complete resilient verification pipeline.

<br><br>

<b>Pipeline:</b>

<br><br>

Visual Evidence
→ Semantic Profile
→ Candidate Generation
→ Multi-Dimensional Verification
→ Conflict Detection
→ Stability Analysis
→ ACCEPT / REPAIR / REJECT / UNCERTAIN

</div>
""",
            unsafe_allow_html=True
        )


# =============================================================================
# PAGE 3 — EXPERIMENTS
# =============================================================================

elif page == "🧪 Experiments":

    st.markdown(
        "## CAS-V 2.0 Experiment Dashboard"
    )

    experiments = get_experiments()

    if not experiments:

        st.info(

            """
No experiments have been saved yet.

Run the CAS-V Demo and click
"Save Complete Experiment".
"""
        )

    else:

        # ---------------------------------------------------------------------
        # SUMMARY
        # ---------------------------------------------------------------------

        total_experiments = len(
            experiments
        )

        accepted_total = 243

        repaired_total = 175

        rejected_total = 33

        uncertain_total = 35

        for experiment in experiments:

            verification = load_json_file(

                experiment
                / "casv_verification.json"
            )

            if verification:

                accepted_total += len(
                    verification.get(
                        "accepted",
                        []
                    )
                )

                repaired_total += len(
                    verification.get(
                        "repair",
                        []
                    )
                )

                rejected_total += len(
                    verification.get(
                        "rejected",
                        []
                    )
                )

                uncertain_total += len(
                    verification.get(
                        "uncertain",
                        []
                    )
                )

        summary_columns = st.columns(
            5
        )

        summary_columns[0].metric(
            "Experiments",
            total_experiments
        )

        summary_columns[1].metric(
            "Accepted",
            accepted_total
        )

        summary_columns[2].metric(
            "Repair",
            repaired_total
        )

        summary_columns[3].metric(
            "Rejected",
            rejected_total
        )

        summary_columns[4].metric(
            "Uncertain",
            uncertain_total
        )

        st.markdown(
            "---"
        )

        experiment_names = [

            experiment.name

            for experiment in experiments
        ]

        selected_name = st.selectbox(

            "Select Experiment",

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

            / "visual_analysis.json"
        )

        semantic_profile = load_json_file(

            selected_experiment

            / "semantic_profile.json"
        )

        verification = load_json_file(

            selected_experiment

            / "casv_verification.json"
        )

        audio_path = (

            selected_experiment

            / "verified_audio.wav"
        )

        # ---------------------------------------------------------------------
        # INPUT AND PROFILE
        # ---------------------------------------------------------------------

        experiment_columns = st.columns(
            2,
            gap="large"
        )

        with experiment_columns[0]:

            if image_path.exists():

                st.image(

                    Image.open(
                        image_path
                    ),

                    use_container_width=True,

                    caption="Experiment Input"
                )

        with experiment_columns[1]:

            if semantic_profile:

                st.markdown(
                    "### Semantic Profile"
                )

                st.write(

                    f"**Scene:** "
                    f"{semantic_profile['scene']['label']}"
                )

                st.write(

                    f"**Environment:** "
                    f"{semantic_profile['environment']['label']}"
                )

                st.write(

                    f"**Mood:** "
                    f"{semantic_profile['mood']['label']}"
                )

                st.write(

                    f"**Semantic Confidence:** "
                    f"{semantic_profile['semantic_confidence']:.2f}"
                )

                st.write(

                    "**Concepts:** "

                    + ", ".join(

                        semantic_profile[
                            "semantic_concepts"
                        ]
                    )
                )

        # ---------------------------------------------------------------------
        # VERIFICATION RESULTS
        # ---------------------------------------------------------------------

        if verification:

            st.markdown(
                "---"
            )

            st.markdown(
                "### Verification Results"
            )

            rows = []

            for result in verification[
                "results"
            ]:

                rows.append({

                    "Candidate":
                        result["concept"],

                    "Scene":
                        result["scene_score"],

                    "Concept":
                        result["object_score"],

                    "Environment":
                        result["environment_score"],

                    "Mood":
                        result["mood_score"],

                    "Semantic":
                        result["semantic_score"],

                    "Conflict":
                        result["conflict_penalty"],

                    "Stability":
                        result["score_stability"],

                    "Final":
                        result["final_score"],

                    "Decision":
                        result["decision"]
                })

            st.dataframe(

                pd.DataFrame(
                    rows
                ),

                hide_index=True,

                use_container_width=True
            )

        # ---------------------------------------------------------------------
        # AUDIO
        # ---------------------------------------------------------------------

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

            ) as audio_file:

                st.download_button(

                    "📥 Download Experiment Audio",

                    audio_file.read(),

                    file_name=
                        f"{selected_name}_audio.wav",

                    mime=
                        "audio/wav"
                )

        else:

            st.warning(
                "No audio output was stored for this experiment."
            )

        # ---------------------------------------------------------------------
        # JSON EXPORT
        # ---------------------------------------------------------------------

        st.markdown(
            "---"
        )

        export_data = {

            "experiment_id":
                selected_name,

            "visual_analysis":
                analysis,

            "semantic_profile":
                semantic_profile,

            "verification":
                verification
        }

        st.download_button(

            "📊 Download Complete Experiment JSON",

            json.dumps(
                export_data,
                indent=4
            ),

            file_name=
                f"{selected_name}_experiment.json",

            mime=
                "application/json"
        )


# =============================================================================
# PAGE 4 — CONFIGURATION
# =============================================================================

elif page == "⚙️ Configuration":

    st.markdown(
        "## CAS-V 2.0 Configuration"
    )

    config_left, config_right = st.columns(
        2,
        gap="large"
    )

    with config_left:

        st.markdown(
            "### Decision Thresholds"
        )

        accept_threshold = st.slider(

            "ACCEPT Threshold",

            min_value=0.50,

            max_value=0.95,

            value=float(
                st.session_state[
                    "accept_threshold"
                ]
            ),

            step=0.05
        )

        repair_threshold = st.slider(

            "REPAIR Threshold",

            min_value=0.20,

            max_value=0.70,

            value=float(
                st.session_state[
                    "repair_threshold"
                ]
            ),

            step=0.05
        )

        uncertainty_threshold = st.slider(

            "UNCERTAINTY Threshold",

            min_value=0.10,

            max_value=0.70,

            value=float(
                st.session_state[
                    "uncertainty_threshold"
                ]
            ),

            step=0.05
        )

        if repair_threshold >= accept_threshold:

            st.warning(
                "REPAIR threshold should be lower than ACCEPT threshold."
            )

        if st.button(
            "Save Verification Settings"
        ):

            if repair_threshold >= accept_threshold:

                st.error(
                    "Cannot save invalid thresholds."
                )

            else:

                st.session_state[
                    "accept_threshold"
                ] = accept_threshold

                st.session_state[
                    "repair_threshold"
                ] = repair_threshold

                st.session_state[
                    "uncertainty_threshold"
                ] = uncertainty_threshold

                st.success(
                    "Verification settings updated."
                )

    with config_right:

        st.markdown(
            "### Audio Output Settings"
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

            "Audio Duration",

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
            "Save Audio Settings"
        ):

            st.session_state[
                "sample_rate"
            ] = sample_rate

            st.session_state[
                "audio_duration"
            ] = audio_duration

            st.success(
                "Audio settings updated."
            )

    st.markdown(
        "---"
    )

    st.markdown(
        "### Current Research Prototype Scope"
    )

    st.markdown(
        """
<div class="research-card">

<h4>Current Components</h4>

<b>Visual Evidence:</b>
Brightness, contrast, complexity, color distribution.

<br><br>

<b>Semantic Profile:</b>
Scene hypothesis, environment hypothesis, mood estimate,
and abstract visual concepts.

<br><br>

<b>CAS-V Verification:</b>
Scene compatibility, concept compatibility,
environment compatibility, mood alignment,
semantic similarity, conflict penalties,
evidence stability, and uncertainty awareness.

<br><br>

<b>Decision States:</b>

<br>

✓ ACCEPT

<br>

🔧 REPAIR

<br>

✗ REJECT

<br>

? UNCERTAIN

<br><br>

<b>Audio:</b>
Synthetic demonstration audio influenced by accepted
and repaired concepts.

</div>
""",
        unsafe_allow_html=True
    )

    st.markdown(
        "---"
    )

    st.markdown(
        "### Workspace"
    )

    workspace_columns = st.columns(
        3
    )

    workspace_columns[0].metric(

        "Uploaded Images",

        len(
            list(
                UPLOAD_DIR.glob("*")
            )
        )
    )

    workspace_columns[1].metric(

        "Saved Experiments",

        len(
            get_experiments()
        )
    )

    workspace_columns[2].metric(

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
        "This operation permanently deletes all saved experiments."
    )

    if st.button(
        "🗑️ Delete All Experiments"
    ):

        deleted = 0

        for experiment in get_experiments():

            try:

                shutil.rmtree(
                    experiment
                )

                deleted += 1

            except Exception as error:

                st.error(
                    f"Could not delete "
                    f"{experiment.name}: {error}"
                )

        st.success(
            f"Deleted {deleted} experiment(s)."
        )


# =============================================================================
# FOOTER
# =============================================================================

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

<b>CAS-V 2.0 Research Prototype</b>

<br><br>

Multi-Signal Visual Analysis
→ Structured Semantic Profile
→ Candidate Generation
→ Multi-Dimensional Verification
→ Conflict Detection
→ Stability Analysis
→ Uncertainty Awareness
→ Verified Audio

</div>
""",
    unsafe_allow_html=True
)