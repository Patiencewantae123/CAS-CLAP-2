import os
import json
import uuid
import shutil
from pathlib import Path
from datetime import datetime

import numpy as np
from PIL import Image
from scipy.io.wavfile import write as wav_write

import discord
from discord.ext import commands

# =============================================================================
# PROJECT DIRECTORIES
# =============================================================================

BASE_DIR = Path.cwd()
UPLOAD_DIR = BASE_DIR / "uploaded_images"
EXPERIMENT_DIR = BASE_DIR / "experiments"
TEMP_DIR = BASE_DIR / "temporary"

UPLOAD_DIR.mkdir(exist_ok=True)
EXPERIMENT_DIR.mkdir(exist_ok=True)
TEMP_DIR.mkdir(exist_ok=True)

# System Thresholds
ACCEPT_THRESHOLD = 0.75
REPAIR_THRESHOLD = 0.45
UNCERTAINTY_THRESHOLD = 0.40

# =============================================================================
# CORE CAS-V ENGINE LOGIC
# =============================================================================

def clamp(value, minimum=0.0, maximum=1.0):
    return max(minimum, min(value, maximum))

def analyze_image(image_path):
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
        scene_candidates.append({"label": "dim / low-light environment", "score": clamp((0.50 - brightness) * 1.5)})
    if brightness > 0.60:
        scene_candidates.append({"label": "bright / open environment", "score": clamp(brightness)})
    if cool_ratio > warm_ratio + 0.08:
        scene_candidates.append({"label": "cool-toned natural or open environment", "score": clamp(cool_ratio)})
    if warm_ratio > cool_ratio + 0.08:
        scene_candidates.append({"label": "warm-toned environment", "score": clamp(warm_ratio)})
    if complexity > 0.55:
        scene_candidates.append({"label": "visually complex environment", "score": complexity})

    if not scene_candidates:
        scene_candidates.append({"label": "general visual environment", "score": 0.45})

    scene_candidates = sorted(scene_candidates, key=lambda item: item["score"], reverse=True)
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

    top_score = primary_scene["score"]
    second_score = scene_candidates[1]["score"] if len(scene_candidates) > 1 else 0.0
    separation = abs(top_score - second_score)
    semantic_confidence = clamp((top_score * 0.65) + (separation * 0.35))

    return {
        "primary_scene": primary_scene,
        "environment": {"label": environment, "confidence": environment_confidence},
        "mood": {"valence": valence, "arousal": arousal},
        "semantic_confidence": semantic_confidence
    }

def build_semantic_profile(analysis):
    scene_label = analysis["primary_scene"]["label"]
    scene_confidence = analysis["primary_scene"]["score"]
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

    return {
        "scene": {"label": scene_label, "confidence": scene_confidence},
        "environment": analysis["environment"],
        "mood": analysis["mood"],
        "semantic_concepts": list(dict.fromkeys(concepts)),
        "semantic_confidence": analysis["semantic_confidence"]
    }

AUDIO_CANDIDATES = [
    {"concept": "Ocean Waves", "scene_tags": ["open", "cool-toned", "nature"], "environment_tags": ["open", "outdoor"], "mood_profile": {"valence": 0.70, "arousal": 0.30}},
    {"concept": "Bird Chirps", "scene_tags": ["open", "nature", "illuminated"], "environment_tags": ["open", "outdoor"], "mood_profile": {"valence": 0.80, "arousal": 0.40}},
    {"concept": "Forest Ambience", "scene_tags": ["nature", "open"], "environment_tags": ["outdoor"], "mood_profile": {"valence": 0.65, "arousal": 0.35}},
    {"concept": "Piano Music", "scene_tags": ["indoor", "performance"], "environment_tags": ["enclosed", "mixed"], "mood_profile": {"valence": 0.65, "arousal": 0.40}},
    {"concept": "Soft Piano", "scene_tags": ["calm", "neutral"], "environment_tags": ["mixed"], "mood_profile": {"valence": 0.65, "arousal": 0.25}},
    {"concept": "Car Engine", "scene_tags": ["urban", "warm-toned", "complex"], "environment_tags": ["open", "mixed"], "mood_profile": {"valence": 0.45, "arousal": 0.70}},
    {"concept": "Traffic Ambience", "scene_tags": ["urban", "complex"], "environment_tags": ["open", "mixed"], "mood_profile": {"valence": 0.40, "arousal": 0.75}}
]

def verify_and_rank(profile):
    scene_text = profile["scene"]["label"].lower()
    env_label = profile["environment"]["label"].lower()
    
    verified_pool = []
    for candidate in AUDIO_CANDIDATES:
        matches = sum(1 for tag in candidate["scene_tags"] if tag in scene_text)
        scene_comp = clamp(matches / max(1, len(candidate["scene_tags"])))
        
        obj_comp = 0.50
        env_comp = clamp(0.30 + (0.40 if any(t in env_label for t in candidate["environment_tags"]) else 0))
        
        mood_dist = (abs(profile["mood"]["valence"] - candidate["mood_profile"]["valence"]) +
                     abs(profile["mood"]["arousal"] - candidate["mood_profile"]["arousal"])) / 2.0
        mood_comp = clamp(1.0 - mood_dist)

        conflicts = []
        if "dark" in env_label and "illuminated" in candidate["scene_tags"]:
            conflicts.append("Low-light image conflicts with illuminated sound tag.")
        
        penalty = len(conflicts) * 0.35
        raw_score = (scene_comp * 0.25) + (obj_comp * 0.15) + (env_comp * 0.20) + (mood_comp * 0.20)
        final_score = clamp(raw_score - penalty)

        if final_score >= ACCEPT_THRESHOLD:
            decision = "ACCEPT"
        elif final_score >= REPAIR_THRESHOLD:
            decision = "REPAIR"
        else:
            decision = "REJECT"

        verified_pool.append({
            "candidate": candidate,
            "scene_comp": scene_comp,
            "obj_comp": obj_comp,
            "env_comp": env_comp,
            "mood_comp": mood_comp,
            "final_score": final_score,
            "decision": decision,
            "conflicts": conflicts
        })

    return sorted(verified_pool, key=lambda x: x["final_score"], reverse=True)

def synthesize_audio(concept_name, duration=4.0, sample_rate=22050):
    time = np.linspace(0, duration, int(sample_rate * duration), False)
    np.random.seed(abs(hash(concept_name)) % (2**32))

    if "Wave" in concept_name or "Ocean" in concept_name:
        envelope = 0.5 + 0.5 * np.sin(2 * np.pi * 0.25 * time)
        audio = np.random.normal(0, 0.2, len(time)) * envelope
    elif "Bird" in concept_name:
        carrier = np.sin(2 * np.pi * 2500 * time + np.sin(2 * np.pi * 12 * time) * 5)
        pulse = (np.sin(2 * np.pi * 3 * time) > 0.7).astype(float)
        audio = carrier * pulse * 0.3
    elif "Piano" in concept_name:
        audio = sum(np.sin(2 * np.pi * f * time) * np.exp(-1.5 * (time % 1.0)) for f in [261.63, 329.63, 392.00]) * 0.25
    else:
        freq = 110.0 + (abs(hash(concept_name)) % 300)
        audio = 0.3 * np.sin(2 * np.pi * freq * time)

    audio = audio / (np.max(np.abs(audio)) + 1e-6)
    pcm = (audio * 32767).astype(np.int16)
    
    out_path = TEMP_DIR / f"prototype_{uuid.uuid4().hex[:8]}.wav"
    wav_write(out_path, sample_rate, pcm)
    return out_path

# =============================================================================
# DISCORD BOT EVENT HANDLERS
# =============================================================================

intents = discord.Intents.default()
intents.message_content = True
bot = commands.Bot(command_prefix="!", intents=intents)

@bot.event
async def on_ready():
    print(f"Logged in as {bot.user.name} ({bot.user.id})")
    print("Send an image with the command !generate or attach an image to trigger analysis.")

@bot.command(name="generate")
async def generate(ctx):
    if not ctx.message.attachments:
        await ctx.send("Please attach an image file when calling `!generate`.")
        return

    attachment = ctx.message.attachments[0]
    if not any(attachment.filename.lower().endswith(ext) for ext in [".png", ".jpg", ".jpeg", ".webp"]):
        await ctx.send("Unsupported image format. Please send PNG, JPG, or WEBP.")
        return

    await ctx.send("🔍 Processing image with CAS-V 2.0...")

    # Save Image
    img_path = UPLOAD_DIR / f"{uuid.uuid4().hex[:8]}_{attachment.filename}"
    await attachment.save(img_path)

    # Run Pipeline
    analysis = analyze_image(img_path)
    profile = build_semantic_profile(analysis)
    results = verify_and_rank(profile)
    top = results[0]

    # Synthesize Audio
    audio_path = synthesize_audio(top["candidate"]["concept"])

    # Create Embed Response
    embed = discord.Embed(
        title="🛡️ CAS-V 2.0 Verification Result",
        color=discord.Color.green() if top["decision"] == "ACCEPT" else discord.Color.gold()
    )
    embed.add_field(name="Primary Scene", value=profile["scene"]["label"].title(), inline=True)
    embed.add_field(name="Environment", value=profile["environment"]["label"].title(), inline=True)
    embed.add_field(name="Confidence", value=f"{profile['semantic_confidence']*100:.1f}%", inline=True)
    
    embed.add_field(name="Selected Sound Concept", value=f"**{top['candidate']['concept']}**", inline=False)
    embed.add_field(name="Verification Score", value=f"`{top['final_score']:.3f}`", inline=True)
    embed.add_field(name="Decision", value=f"**{top['decision']}**", inline=True)

    if top["conflicts"]:
        embed.add_field(name="⚠️ Explicit Conflicts", value="\n".join(f"- {c}" for c in top["conflicts"]), inline=False)

    await ctx.send(embed=embed)
    await ctx.send(file=discord.File(audio_path, filename="generated_audio.wav"))

# Replace with your actual Discord Bot Token
BOT_TOKEN = "YOUR_DISCORD_BOT_TOKEN_HERE"

if __name__ == "__main__":
    bot.run(BOT_TOKEN)
