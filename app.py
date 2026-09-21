# =============================================================================
# CAS-V 2.0
# Conflict-Aware Semantic Verification for Reliable Image-to-Audio Generation
#
# Research Prototype
# =============================================================================

import os
import json
import uuid
import shutil
import sys

from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd
import streamlit as st
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
# 4. HELPER FUNCTIONS
# =============================================================================

def clamp(value, minimum=0.0, maximum=1.0):
    return max(minimum, min(value, maximum))


def safe_filename(name):
    return "".join(
        character if (character.isalnum() or character in "._-") else "_"
        for character in name
    )


def save_uploaded_image(uploaded_file):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    unique_id = uuid.uuid4().hex[:8]
    filename = f"{timestamp}_{unique_id}_{safe_filename(uploaded_file.name)}"
    image_path = UPLOAD_DIR / filename

    with open(image_path, "wb") as file:
        file.write(uploaded_file.getbuffer())

    return image_path


# =============================================================================
# 5. MULTI-SIGNAL VISUAL ANALYSIS
# =============================================================================

def analyze_uploaded_image(image_path):
    image = Image.open(image_path).convert("RGB")
    array = np.array(image).astype(np.float32) / 255.0

    brightness = float(array.mean())
    contrast = float(array.std())
    red_mean = float(array[:, :, 0].mean())
    green_mean = float(array[:, :, 1].mean())
    blue_mean = float(array[:, :, 2].mean())

    color_values = {"red": red_mean, "green": green_mean, "blue": blue_mean}
    dominant_color = max(color_values, key=color_values.get)

    warm_signal = (red_mean + 0.5 * green_mean) / 1.5
    cool_signal = (blue_mean + 0.5 * green_mean) / 1.5
    warm_ratio = clamp(warm_signal)
    cool_ratio = clamp(cool_signal)

    complexity = clamp(contrast * 2.5)

    scene_candidates = []
    if brightness < 0.35:
        scene_candidates.append({
            "label": "dim / low-light environment",
            "score": clamp((0.50 - brightness) * 1.5)
        })
    if brightness > 0.60:
        scene_candidates.append({
            "label": "bright / open environment",
            "score": clamp(brightness)
        })
    if cool_ratio > warm_ratio + 0.08:
        scene_candidates.append({
            "label": "cool-toned natural or open environment",
            "score": clamp(cool_ratio)
        })
    if warm_ratio > cool_ratio + 0.08:
        scene_candidates.append({
            "label": "warm-toned environment",
            "score": clamp(warm_ratio)
        })
    if complexity > 0.55:
        scene_candidates.append({
            "label": "visually complex environment",
            "score": complexity
        })

    if not scene_candidates:
        scene_candidates.append({
            "label": "general visual environment",
            "score": 0.45
        })

    scene_candidates = sorted(
        scene_candidates,
        key=lambda item: item["score"],
        reverse=True
    )
    primary_scene = scene_candidates[0]

    if brightness > 0.60:
        environment = "open / illuminated"
        environment_confidence = clamp(brightness)
    elif brightness < 0.35:
        environment = "dark / enclosed"
        environment_confidence = clamp(1.0 - brightness)
    else:
        environment = "mixed / uncertain"
        environment_confidence = 0.50

    valence = clamp(0.45 + (warm_ratio * 0.30) + (brightness * 0.25) - (contrast * 0.10))
    arousal = clamp(0.25 + (contrast * 0.60) + (complexity * 0.25))

    if valence >= 0.65 and arousal < 0.55:
        mood_label = "calm / positive"
    elif arousal >= 0.65:
        mood_label = "energetic / intense"
    elif valence < 0.40:
        mood_label = "dark / subdued"
    else:
        mood_label = "neutral / mixed"

    top_score = primary_scene["score"]
    second_score = scene_candidates[1]["score"] if len(scene_candidates) > 1 else 0.0
    separation = abs(top_score - second_score)
    semantic_confidence = clamp((top_score * 0.65) + (separation * 0.35))

    return {
        "visual_features": {
            "brightness": brightness,
            "contrast": contrast,
            "complexity": complexity,
            "dominant_color": dominant_color,
            "red_mean": red_mean,
            "green_mean": green_mean,
            "blue_mean": blue_mean,
            "warm_ratio": warm_ratio,
            "cool_ratio": cool_ratio
        },
        "scene_candidates": scene_candidates,
        "primary_scene": primary_scene,
        "environment": {
            "label": environment,
            "confidence": environment_confidence
        },
        "mood": {
            "valence": valence,
            "arousal": arousal,
            "label": mood_label
        },
        "semantic_confidence": semantic_confidence,
        "analysis_method": "Multi-signal visual statistics and heuristic semantic profiling"
    }


# =============================================================================
# 6. STRUCTURED SEMANTIC PROFILE
# =============================================================================

def build_semantic_profile(analysis):
    scene_label = analysis["primary_scene"]["label"]
    scene_confidence = analysis["primary_scene"]["score"]
    environment = analysis["environment"]
    mood = analysis["mood"]

    concepts = []
    scene_lower = scene_label.lower()

    if "cool" in scene_lower:
        concepts.extend(["open", "cool-toned", "possible nature"])
    if "warm" in scene_lower:
        concepts.extend(["warm-toned", "possible activity"])
    if "bright" in scene_lower:
        concepts.extend(["illuminated", "open"])
    if "dim" in scene_lower:
        concepts.extend(["low-light", "enclosed"])
    if analysis["visual_features"]["complexity"] > 0.55:
        concepts.append("complex visual structure")

    concepts = list(dict.fromkeys(concepts))

    return {
        "scene": {
            "label": scene_label,
            "confidence": scene_confidence
        },
        "environment": environment,
        "mood": mood,
        "semantic_concepts": concepts,
        "semantic_confidence": analysis["semantic_confidence"],
        "source_method": analysis["analysis_method"]
    }


# =============================================================================
# 7. AUDIO CANDIDATE KNOWLEDGE BASE
# =============================================================================

AUDIO_CANDIDATES = [
    {
        "concept": "Ocean Waves",
        "category": "natural",
        "scene_tags": ["open", "cool-toned", "nature"],
        "environment_tags": ["open", "outdoor"],
        "mood_profile": {"valence": 0.70, "arousal": 0.30}
    },
    {
        "concept": "Bird Chirps",
        "category": "natural",
        "scene_tags": ["open", "nature", "illuminated"],
        "environment_tags": ["open", "outdoor"],
        "mood_profile": {"valence": 0.80, "arousal": 0.40}
    },
    {
        "concept": "Forest Ambience",
        "category": "natural",
        "scene_tags": ["nature", "open"],
        "environment_tags": ["outdoor"],
        "mood_profile": {"valence": 0.65, "arousal": 0.35}
    },
    {
        "concept": "Piano Music",
        "category": "music",
        "scene_tags": ["indoor", "performance"],
        "environment_tags": ["enclosed", "mixed"],
        "mood_profile": {"valence": 0.65, "arousal": 0.40}
    },
    {
        "concept": "Soft Piano",
        "category": "ambient",
        "scene_tags": ["calm", "neutral"],
        "environment_tags": ["mixed"],
        "mood_profile": {"valence": 0.65, "arousal": 0.25}
    },
    {
        "concept": "Ambient Synth",
        "category": "ambient",
        "scene_tags": ["abstract", "mixed"],
        "environment_tags": ["mixed"],
        "mood_profile": {"valence": 0.55, "arousal": 0.35}
    },
    {
        "concept": "Car Engine",
        "category": "urban",
        "scene_tags": ["urban", "warm-toned", "complex"],
        "environment_tags": ["open", "mixed"],
        "mood_profile": {"valence": 0.45, "arousal": 0.70}
    },
    {
        "concept": "Traffic Ambience",
        "category": "urban",
        "scene_tags": ["urban", "complex"],
        "environment_tags": ["open", "mixed"],
        "mood_profile": {"valence": 0.40, "arousal": 0.75}
    },
    {
        "concept": "Audience Applause",
        "category": "performance",
        "scene_tags": ["performance", "complex"],
        "environment_tags": ["enclosed", "mixed"],
        "mood_profile": {"valence": 0.75, "arousal": 0.85}
    },
    {
        "concept": "Heavy Concert",
        "category": "performance",
        "scene_tags": ["performance", "dark", "complex"],
        "environment_tags": ["enclosed"],
        "mood_profile": {"valence": 0.45, "arousal": 0.95}
    }
]


# =============================================================================
# 8. CANDIDATE GENERATION
# =============================================================================

def generate_audio_candidates(semantic_profile, top_k=8):
    candidates = []
    scene_text = semantic_profile["scene"]["label"].lower()
    mood = semantic_profile["mood"]
    concepts = semantic_profile["semantic_concepts"]

    for item in AUDIO_CANDIDATES:
        relevance = 0.35
        for tag in item["scene_tags"]:
            if tag in scene_text:
                relevance += 0.12
            for concept in concepts:
                if tag in concept:
                    relevance += 0.08

        candidate_mood = item["mood_profile"]
        mood_distance = (
            abs(mood["valence"] - candidate_mood["valence"]) +
            abs(mood["arousal"] - candidate_mood["arousal"])
        ) / 2
        mood_similarity = 1.0 - mood_distance
        relevance += mood_similarity * 0.20

        relevance = clamp(relevance)

        candidates.append({
            "concept": item["concept"],
            "category": item["category"],
            "preliminary_score": relevance,
            "scene_tags": item["scene_tags"],
            "environment_tags": item["environment_tags"],
            "mood_profile": item["mood_profile"]
        })

    candidates = sorted(candidates, key=lambda item: item["preliminary_score"], reverse=True)
    return candidates[:top_k]


# =============================================================================
# 9. CAS-V VERIFICATION MODULES
# =============================================================================

def calculate_scene_compatibility(semantic_profile, candidate):
    scene_text = semantic_profile["scene"]["label"].lower()
    semantic_concepts = semantic_profile["semantic_concepts"]
    matches = 0
    possible = max(1, len(candidate["scene_tags"]))

    for tag in candidate["scene_tags"]:
        if tag in scene_text:
            matches += 1
            continue
        for concept in semantic_concepts:
            if tag in concept:
                matches += 1
                break

    return clamp(matches / possible)


def calculate_object_compatibility(semantic_profile, candidate):
    concepts = semantic_profile["semantic_concepts"]
    candidate_tags = candidate["scene_tags"] + candidate["environment_tags"]
    if not concepts:
        return 0.50

    matches = 0
    for concept in concepts:
        for tag in candidate_tags:
            if tag in concept or concept in tag:
                matches += 1
                break

    return clamp(matches / len(concepts))


def calculate_environment_compatibility(semantic_profile, candidate):
    environment_label = semantic_profile["environment"]["label"].lower()
    environment_confidence = semantic_profile["environment"]["confidence"]

    score = 0.30
    for tag in candidate["environment_tags"]:
        if tag in environment_label:
            score += 0.40
        if "open" in environment_label and tag == "outdoor":
            score += 0.20
        if "dark" in environment_label and tag == "enclosed":
            score += 0.20
        if "mixed" in environment_label and tag == "mixed":
            score += 0.25

    return clamp(score * environment_confidence)


def calculate_mood_compatibility(semantic_profile, candidate):
    image_mood = semantic_profile["mood"]
    candidate_mood = candidate["mood_profile"]

    valence_distance = abs(image_mood["valence"] - candidate_mood["valence"])
    arousal_distance = abs(image_mood["arousal"] - candidate_mood["arousal"])
    distance = (valence_distance + arousal_distance) / 2.0
    return clamp(1.0 - distance)


def detect_explicit_conflicts(semantic_profile, candidate):
    conflicts = []
    scene_text = semantic_profile["scene"]["label"].lower()
    env_label = semantic_profile["environment"]["label"].lower()

    if "dark" in env_label and "illuminated" in candidate["scene_tags"]:
        conflicts.append("Low-light image conflicts with illuminated candidate sound tag.")

    if "cool-toned" in scene_text and "warm-toned" in candidate["scene_tags"]:
        conflicts.append("Cool-toned visual scene conflicts with warm-toned sound profile.")

    conflict_penalty = len(conflicts) * 0.35
    return conflicts, conflict_penalty


def verify_candidates_casv(semantic_profile, candidates):
    verified_pool = []

    for candidate in candidates:
        scene_comp = calculate_scene_compatibility(semantic_profile, candidate)
        obj_comp = calculate_object_compatibility(semantic_profile, candidate)
        env_comp = calculate_environment_compatibility(semantic_profile, candidate)
        mood_comp = calculate_mood_compatibility(semantic_profile, candidate)

        semantic_sim = (scene_comp * 0.4) + (obj_comp * 0.3) + (env_comp * 0.3)
        conflicts, penalty = detect_explicit_conflicts(semantic_profile, candidate)

        raw_score = (
            (scene_comp * 0.25) +
            (obj_comp * 0.15) +
            (env_comp * 0.20) +
            (mood_comp * 0.20) +
            (semantic_sim * 0.20)
        )

        final_score = clamp(raw_score - penalty)
        uncertainty = clamp(1.0 - semantic_profile["semantic_confidence"])

        if final_score >= st.session_state["accept_threshold"]:
            decision = "ACCEPT"
        elif final_score >= st.session_state["repair_threshold"]:
            decision = "REPAIR"
        elif uncertainty >= st.session_state["uncertainty_threshold"]:
            decision = "UNCERTAIN"
        else:
            decision = "REJECT"

        verified_pool.append({
            "candidate": candidate,
            "scene_comp": scene_comp,
            "obj_comp": obj_comp,
            "env_comp": env_comp,
            "mood_comp": mood_comp,
            "semantic_sim": semantic_sim,
            "conflicts": conflicts,
            "penalty": penalty,
            "final_score": final_score,
            "uncertainty": uncertainty,
            "decision": decision
        })

    return sorted(verified_pool, key=lambda item: item["final_score"], reverse=True)


# =============================================================================
# 10. AUDIO SYNTHESIS & EXPERIMENT LOGGING
# =============================================================================

def synthesize_prototype_audio(concept_name, duration=4.0, sample_rate=22050):
    time = np.linspace(0, duration, int(sample_rate * duration), False)
    np.random.seed(abs(hash(concept_name)) % (2**32))

    if "Wave" in concept_name or "Ocean" in concept_name:
        envelope = 0.5 + 0.5 * np.sin(2 * np.pi * 0.25 * time)
        noise = np.random.normal(0, 0.2, len(time))
        audio = noise * envelope
    elif "Bird" in concept_name:
        carrier = np.sin(2 * np.pi * 2500 * time + np.sin(2 * np.pi * 12 * time) * 5)
        pulse = (np.sin(2 * np.pi * 3 * time) > 0.7).astype(float)
        audio = carrier * pulse * 0.3
    elif "Piano" in concept_name:
        freqs = [261.63, 329.63, 392.00]
        audio = np.zeros_like(time)
        for freq in freqs:
            decay = np.exp(-1.5 * (time % 1.0))
            audio += np.sin(2 * np.pi * freq * time) * decay
        audio *= 0.25
    else:
        freq = 110.0 + (abs(hash(concept_name)) % 300)
        audio = 0.3 * np.sin(2 * np.pi * freq * time)

    audio = audio / (np.max(np.abs(audio)) + 1e-6)
    pcm = (audio * 32767).astype(np.int16)

    output_path = TEMP_DIR / f"prototype_{uuid.uuid4().hex[:8]}.wav"
    wav_write(output_path, sample_rate, pcm)
    return output_path


def log_experiment(image_path, semantic_profile, verified_results, selected_result, audio_path):
    exp_id = f"exp_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:6]}"
    exp_folder = EXPERIMENT_DIR / exp_id
    exp_folder.mkdir(exist_ok=True)

    dest_image = exp_folder / Path(image_path).name
    shutil.copy(image_path, dest_image)

    dest_audio = exp_folder / Path(audio_path).name
    shutil.copy(audio_path, dest_audio)

    meta = {
        "experiment_id": exp_id,
        "timestamp": datetime.now().isoformat(),
        "semantic_profile": semantic_profile,
        "selected_concept": selected_result["candidate"]["concept"],
        "verification_details": {
            "final_score": selected_result["final_score"],
            "decision": selected_result["decision"],
            "conflicts": selected_result["conflicts"]
        }
    }

    with open(exp_folder / "experiment_meta.json", "w") as file:
        json.dump(meta, file, indent=4)

    return exp_id


# =============================================================================
# 11. STREAMLIT APPLICATION DASHBOARD
# =============================================================================

st.title("🛡️ CAS-V 2.0 Research Interface")
st.caption("Conflict-Aware Semantic Verification System for Image-to-Audio Reliability")

with st.sidebar:
    st.header("⚙️ System Configuration")
    st.session_state["accept_threshold"] = st.slider("ACCEPT Threshold", 0.50, 0.95, st.session_state["accept_threshold"], 0.05)
    st.session_state["repair_threshold"] = st.slider("REPAIR Threshold", 0.20, 0.70, st.session_state["repair_threshold"], 0.05)
    st.session_state["uncertainty_threshold"] = st.slider("Uncertainty Tolerance", 0.10, 0.80, st.session_state["uncertainty_threshold"], 0.05)

uploaded_file = st.file_uploader("Upload Image Target for Analysis", type=["png", "jpg", "jpeg", "webp"])

if uploaded_file is not None:
    image_path = save_uploaded_image(uploaded_file)
    st.session_state["last_image_path"] = image_path

    st.subheader("Visual Source Data")
    col_img, col_metrics = st.columns([1, 2])

    with col_img:
        st.image(str(image_path), use_container_width=True, caption=uploaded_file.name)

    with col_metrics:
        with st.spinner("Analyzing image features..."):
            analysis = analyze_uploaded_image(image_path)
            profile = build_semantic_profile(analysis)
            candidates = generate_audio_candidates(profile)
            verified = verify_candidates_casv(profile, candidates)

            st.session_state["last_analysis"] = analysis
            st.session_state["last_semantic_profile"] = profile
            st.session_state["last_candidates"] = candidates
            st.session_state["last_verification"] = verified

        m1, m2, m3 = st.columns(3)
        m1.metric("Primary Scene", profile["scene"]["label"].title())
        m2.metric("Environment", profile["environment"]["label"].title())
        m3.metric("Semantic Confidence", f"{profile['semantic_confidence'] * 100:.1f}%")

    st.divider()
    st.subheader("CAS-V Verification Pipeline Output")

    top_verified = verified[0]
    candidate = top_verified["candidate"]

    d_col, info_col = st.columns([1, 2])
    with d_col:
        st.write("### Recommended Action")
        decision = top_verified["decision"]
        if decision == "ACCEPT":
            st.success(f"Status: **{decision}**")
        elif decision == "REPAIR":
            st.warning(f"Status: **{decision}**")
        elif decision == "UNCERTAIN":
            st.info(f"Status: **{decision}**")
        else:
            st.error(f"Status: **{decision}**")

        st.metric("Top Candidate", candidate["concept"])
        st.metric("Final Verification Score", f"{top_verified['final_score']:.3f}")

    with info_col:
        st.write("### Dimensional Breakdown")
        b1, b2, b3, b4 = st.columns(4)
        b1.metric("Scene Comp.", f"{top_verified['scene_comp']:.2f}")
        b2.metric("Object Comp.", f"{top_verified['obj_comp']:.2f}")
        b3.metric("Env Comp.", f"{top_verified['env_comp']:.2f}")
        b4.metric("Mood Comp.", f"{top_verified['mood_comp']:.2f}")

        if top_verified["conflicts"]:
            st.error("Explicit Conflicts Identified:")
            for item in top_verified["conflicts"]:
                st.write(f"- {item}")
        else:
            st.info("No explicit structural conflicts detected.")

    st.divider()
    st.subheader("Audio Synthesis & Experiment Persistence")

    if st.button("Synthesize Audio & Log Experiment", type="primary"):
        with st.spinner("Synthesizing waveform..."):
            audio_path = synthesize_prototype_audio(candidate["concept"])
            exp_id = log_experiment(image_path, profile, verified, top_verified, audio_path)
            st.session_state["last_audio_path"] = audio_path
            st.session_state["last_experiment_id"] = exp_id

    if st.session_state["last_audio_path"] is not None:
        st.audio(str(st.session_state["last_audio_path"]), format="audio/wav")
        st.success(f"Experiment logged under ID: `{st.session_state['last_experiment_id']}`")
