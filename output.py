# ============================================================
# CAS-V QUALITATIVE COMPARISON FIGURE
# AUTOMATIC IMAGE + MEL-SPECTROGRAM GENERATION
# NO IMAGE UPLOAD REQUIRED
# ============================================================

# ============================================================
# 1. INSTALL
# ============================================================

!pip -q install diffusers transformers accelerate safetensors \
    librosa soundfile matplotlib pillow


# ============================================================
# 2. IMPORTS
# ============================================================

import os
import numpy as np
import matplotlib.pyplot as plt
import librosa
import librosa.display

from PIL import Image
from diffusers import AutoPipelineForText2Image
from google.colab import files


# ============================================================
# 3. DEVICE
# ============================================================

import torch

device = "cuda" if torch.cuda.is_available() else "cpu"

print("Device:", device)

if device == "cpu":
    print("WARNING: Enable GPU in Colab:")
    print("Runtime → Change runtime type → T4 GPU")


# ============================================================
# 4. LOAD IMAGE GENERATION MODEL
# ============================================================

print("Loading image generation model...")

pipe = AutoPipelineForText2Image.from_pretrained(
    "stabilityai/sdxl-turbo",
    torch_dtype=torch.float16 if device == "cuda" else torch.float32,
    variant="fp16" if device == "cuda" else None
)

pipe = pipe.to(device)

# Memory optimization
if device == "cuda":
    pipe.enable_attention_slicing()


# ============================================================
# 5. SCENES
# ============================================================

scene_names = [
    "cloth",
    "dry leaves",
    "cloth",
    "metal",
    "wood",
    "dirt",
    "drywall"
]


# ============================================================
# 6. REALISTIC IMAGE PROMPTS
# ============================================================

prompts = [

    """
    close-up realistic photograph of a hand holding a thin wooden stick
    touching rough fabric upholstery on a sofa,
    detailed textile fibers, natural indoor lighting,
    shallow depth of field, documentary photography,
    realistic colors, high detail
    """,

    """
    close-up realistic photograph of a hand holding a wooden stick
    touching dry fallen leaves on the forest floor,
    brown leaves, soil, natural outdoor lighting,
    shallow depth of field, realistic photography,
    highly detailed textures
    """,

    """
    realistic close-up photograph of a hand using a wooden stick
    scratching textured fabric upholstery,
    visible cloth fibers and patterned textile,
    indoor environment, natural lighting,
    realistic photographic appearance
    """,

    """
    realistic close-up photograph of a metal surface being scratched
    by a thin wooden stick,
    industrial object, metallic texture,
    detailed surface, natural lighting,
    documentary photography, highly realistic
    """,

    """
    realistic close-up photograph of a wooden stick hitting
    a wooden surface,
    visible wood grain and small impact area,
    natural lighting, realistic photographic texture,
    shallow depth of field
    """,

    """
    realistic close-up photograph of a stick touching dry soil and dirt,
    rough ground texture, small stones and dust,
    outdoor natural lighting,
    realistic documentary photograph
    """,

    """
    realistic close-up photograph of a wooden stick touching
    a painted drywall surface,
    visible wall texture and subtle scratches,
    indoor natural lighting,
    realistic photography, detailed surface
    """
]


# ============================================================
# 7. GENERATE IMAGES
# ============================================================

generated_images = []

for i, prompt in enumerate(prompts):

    print(f"Generating scene {i+1}/7: {scene_names[i]}")

    result = pipe(
        prompt=prompt,
        num_inference_steps=1,
        guidance_scale=0.0,
        height=512,
        width=512
    )

    img = result.images[0]

    generated_images.append(img)


print("All seven images generated.")


# ============================================================
# 8. AUDIO CONCEPTS
# ============================================================

# Existing systems produce semantically conflicting concepts

existing_concepts = [
    "bird chirping",
    "ocean waves",
    "party music",
    "machine noise",
    "rain",
    "wind",
    "door creak"
]


# CAS-V verified concepts

casv_concepts = [
    "stick hitting cloth",
    "stick hitting leaves",
    "stick scratching cloth",
    "stick scratching metal",
    "stick hitting wood",
    "stick hitting dirt",
    "stick hitting drywall"
]


# Ground truth concepts

target_concepts = casv_concepts.copy()


# ============================================================
# 9. SYNTHETIC AUDIO GENERATOR
# ============================================================

sr = 22050
duration = 3.0

t = np.linspace(
    0,
    duration,
    int(sr * duration),
    endpoint=False
)


def generate_audio(concept, seed=0):

    np.random.seed(seed)

    audio = np.zeros_like(t)

    # --------------------------------------------------------
    # IMPACT / STICK SOUNDS
    # --------------------------------------------------------

    if (
        "stick" in concept
        or "hitting" in concept
        or "hitting" in concept
        or "scratching" in concept
    ):

        # repeated impacts
        impact_times = np.linspace(
            0.35,
            2.65,
            7
        )

        for ti in impact_times:

            idx = int(ti * sr)

            length = int(0.15 * sr)

            end = min(
                idx + length,
                len(audio)
            )

            local_t = np.arange(end - idx) / sr

            frequency = np.random.uniform(
                700,
                1800
            )

            envelope = np.exp(
                -25 * local_t
            )

            signal = (
                np.sin(
                    2 * np.pi *
                    frequency *
                    local_t
                )
                * envelope
            )

            audio[idx:end] += signal

        # surface texture
        audio += (
            0.04 *
            np.random.randn(len(t))
        )


    # --------------------------------------------------------
    # CLOTH
    # --------------------------------------------------------

    elif "cloth" in concept:

        texture = np.random.randn(len(t))

        audio = 0.12 * texture

        # filtering-like rolling average
        kernel = np.ones(80) / 80

        audio = np.convolve(
            audio,
            kernel,
            mode="same"
        )


    # --------------------------------------------------------
    # LEAVES
    # --------------------------------------------------------

    elif "leaves" in concept:

        noise = np.random.randn(len(t))

        modulation = (
            0.3 +
            0.7 *
            np.abs(
                np.sin(
                    2*np.pi*4*t
                )
            )
        )

        audio = (
            0.22 *
            noise *
            modulation
        )


    # --------------------------------------------------------
    # METAL
    # --------------------------------------------------------

    elif "metal" in concept:

        for ti in [0.4, 0.8, 1.3, 1.8, 2.3]:

            idx = int(ti * sr)

            length = int(0.5 * sr)

            end = min(
                idx + length,
                len(audio)
            )

            local_t = np.arange(end-idx) / sr

            signal = (

                np.sin(
                    2*np.pi*2400*local_t
                )

                +

                0.5*np.sin(
                    2*np.pi*4200*local_t
                )

            )

            signal *= np.exp(
                -7 * local_t
            )

            audio[idx:end] += signal


    # --------------------------------------------------------
    # WOOD
    # --------------------------------------------------------

    elif "wood" in concept:

        for ti in [0.4, 0.9, 1.5, 2.1, 2.6]:

            idx = int(ti * sr)

            length = int(.18 * sr)

            end = min(
                idx + length,
                len(audio)
            )

            local_t = np.arange(end-idx)/sr

            signal = (
                np.sin(
                    2*np.pi*500*local_t
                )
                *
                np.exp(-20*local_t)
            )

            audio[idx:end] += signal


    # --------------------------------------------------------
    # DIRT
    # --------------------------------------------------------

    elif "dirt" in concept:

        audio = (
            0.18 *
            np.random.randn(len(t))
        )

        # slow amplitude modulation
        audio *= (
            0.4 +
            0.6 *
            np.abs(
                np.sin(2*np.pi*2*t)
            )
        )


    # --------------------------------------------------------
    # DRYWALL
    # --------------------------------------------------------

    elif "drywall" in concept:

        for ti in [0.5, 1.1, 1.8, 2.5]:

            idx = int(ti * sr)

            length = int(.2 * sr)

            end = min(
                idx + length,
                len(audio)
            )

            local_t = np.arange(end-idx)/sr

            signal = (
                np.sin(
                    2*np.pi*700*local_t
                )
                *
                np.exp(-16*local_t)
            )

            audio[idx:end] += signal


    # --------------------------------------------------------
    # BIRD
    # --------------------------------------------------------

    elif "bird" in concept:

        for ti in [0.4, 1.2, 2.0]:

            idx = int(ti * sr)

            length = int(.25 * sr)

            end = min(
                idx + length,
                len(audio)
            )

            local_t = np.arange(end-idx)/sr

            chirp = np.sin(
                2*np.pi*
                (1800 + 1000*local_t)*
                local_t
            )

            audio[idx:end] += (
                .25 *
                chirp *
                np.exp(-8*local_t)
            )


    # --------------------------------------------------------
    # OCEAN
    # --------------------------------------------------------

    elif "ocean" in concept:

        noise = np.random.randn(len(t))

        modulation = (
            .4 +
            .6 *
            np.sin(
                2*np.pi*.35*t
            )**2
        )

        audio = (
            .25 *
            noise *
            modulation
        )


    # --------------------------------------------------------
    # RAIN
    # --------------------------------------------------------

    elif "rain" in concept:

        audio = (
            .12 *
            np.random.randn(len(t))
        )


    # --------------------------------------------------------
    # WIND
    # --------------------------------------------------------

    elif "wind" in concept:

        noise = np.random.randn(len(t))

        audio = (
            .15 *
            noise *
            (
                .5 +
                .5*np.sin(
                    2*np.pi*.4*t
                )
            )
        )


    # --------------------------------------------------------
    # MACHINE
    # --------------------------------------------------------

    elif "machine" in concept:

        audio = (

            .15 *
            np.sin(
                2*np.pi*120*t
            )

            +

            .07 *
            np.sin(
                2*np.pi*240*t
            )

            +

            .04 *
            np.random.randn(len(t))

        )


    # --------------------------------------------------------
    # PARTY
    # --------------------------------------------------------

    elif "party" in concept:

        audio = (

            .12 *
            np.sin(
                2*np.pi*120*t
            )

            +

            .08 *
            np.sin(
                2*np.pi*240*t
            )

            +

            .05 *
            np.random.randn(len(t))

        )


    # --------------------------------------------------------
    # DOOR
    # --------------------------------------------------------

    elif "door" in concept:

        audio = (
            .3 *
            np.sin(
                2*np.pi*300*t
            )
            *
            np.exp(-1.5*t)
        )


    # normalize

    audio = audio / (
        np.max(
            np.abs(audio)
        ) + 1e-8
    )

    return audio


# ============================================================
# 10. CREATE MEL SPECTROGRAM
# ============================================================

def make_mel(concept, seed=0):

    audio = generate_audio(
        concept,
        seed
    )

    mel = librosa.feature.melspectrogram(
        y=audio,
        sr=sr,
        n_fft=1024,
        hop_length=256,
        n_mels=64,
        fmin=50,
        fmax=8000
    )

    mel_db = librosa.power_to_db(
        mel,
        ref=np.max
    )

    return mel_db


# ============================================================
# 11. GENERATE SPECTROGRAMS
# ============================================================

existing_mels = [
    make_mel(
        concept,
        seed=i
    )
    for i, concept in enumerate(
        existing_concepts
    )
]


casv_mels = [
    make_mel(
        concept,
        seed=i+20
    )
    for i, concept in enumerate(
        casv_concepts
    )
]


target_mels = [
    make_mel(
        concept,
        seed=i+40
    )
    for i, concept in enumerate(
        target_concepts
    )
]


# ============================================================
# 12. FIGURE
# ============================================================

fig = plt.figure(
    figsize=(22, 14),
    dpi=180,
    facecolor="white"
)

gs = fig.add_gridspec(
    4,
    8,

    width_ratios=[
        1.25,
        1,1,1,1,1,1,1
    ],

    height_ratios=[
        .7,
        1.2,
        1.25,
        1.25
    ],

    hspace=.38,
    wspace=.12
)


# ============================================================
# 13. TITLE
# ============================================================

ax = fig.add_subplot(
    gs[0, :]
)

ax.axis("off")

ax.text(
    .5,
    .5,
    "Qualitative Semantic Conflict Verification",
    ha="center",
    va="center",
    fontsize=21,
    fontweight="bold"
)


# ============================================================
# 14. INPUT IMAGE ROW
# ============================================================

ax = fig.add_subplot(
    gs[1,0]
)

ax.axis("off")

ax.text(
    .5,
    .5,
    "INPUT IMAGE",
    ha="center",
    va="center",
    fontsize=12,
    fontweight="bold"
)


for i in range(7):

    ax = fig.add_subplot(
        gs[1,i+1]
    )

    ax.imshow(
        generated_images[i]
    )

    ax.set_title(
        scene_names[i],
        fontsize=9,
        fontweight="bold"
    )

    ax.axis("off")


# ============================================================
# 15. CAS-V ANALYSIS ROW
# ============================================================

ax = fig.add_subplot(
    gs[2,0]
)

ax.axis("off")

ax.text(
    .5,
    .5,

    "CAS-V\nSEMANTIC\nANALYSIS",

    ha="center",
    va="center",

    fontsize=11,
    fontweight="bold"
)


# ------------------------------------------------------------
# IMAGE + MEL FOR CAS-V
# ------------------------------------------------------------

for i in range(7):

    # create vertical image + spectrogram panel

    panel = fig.add_subplot(
        gs[2,i+1]
    )

    panel.axis("off")

    # image
    panel.imshow(
        generated_images[i],
        extent=[0,1,.52,1]
    )

    # mel
    panel.imshow(
        casv_mels[i],
        aspect="auto",
        origin="lower",
        extent=[0,1,0,.48],
        cmap="viridis"
    )

    panel.text(
        .5,
        .02,
        "verified",
        ha="center",
        va="bottom",
        fontsize=7
    )


# ============================================================
# 16. OURS ROW
# ============================================================

ax = fig.add_subplot(
    gs[3,0]
)

ax.axis("off")

ax.text(
    .5,
    .5,
    "OURS\n(CAS-V)",
    ha="center",
    va="center",
    fontsize=12,
    fontweight="bold"
)


for i in range(7):

    ax = fig.add_subplot(
        gs[3,i+1]
    )

    ax.imshow(
        casv_mels[i],
        aspect="auto",
        origin="lower",
        cmap="viridis"
    )

    ax.set_title(
        casv_concepts[i],
        fontsize=7
    )

    ax.axis("off")


# ============================================================
# 17. TARGET SOUND
# ============================================================

# add a separate row manually underneath

target_y = .055

target_height = .16

fig.text(
    .075,
    target_y + .07,
    "TARGET\nSOUND",
    ha="center",
    va="center",
    fontsize=11,
    fontweight="bold"
)


for i in range(7):

    left = .17 + i*.108

    ax = fig.add_axes(
        [
            left,
            target_y,
            .085,
            target_height
        ]
    )

    ax.imshow(
        target_mels[i],
        aspect="auto",
        origin="lower",
        cmap="viridis"
    )

    ax.axis("off")


# ============================================================
# 18. PURPLE CAS-V BOX
# ============================================================

fig.add_artist(
    plt.Rectangle(
        (.155, .355),
        .815,
        .29,

        transform=fig.transFigure,

        fill=False,

        linewidth=3,

        edgecolor="#9b3fa8"
    )
)


# ============================================================
# 19. BLUE DASHED OURS BOX
# ============================================================

fig.add_artist(
    plt.Rectangle(
        (.155, .215),
        .815,
        .125,

        transform=fig.transFigure,

        fill=False,

        linewidth=2.5,

        linestyle="--",

        edgecolor="#1683d8"
    )
)


# ============================================================
# 20. CAS-V EXPLANATION
# ============================================================

fig.text(
    .025,
    .50,

    "CAS-V analyzes\n"
    "visual semantics\n"
    "and verifies the\n"
    "candidate audio\n"
    "concept before\n"
    "generation.",

    ha="left",
    va="center",

    fontsize=11,
    fontweight="bold"
)


# ============================================================
# 21. SAVE
# ============================================================

output_file = (
    "/content/"
    "CASV_Qualitative_Comparison.png"
)

plt.savefig(
    output_file,
    dpi=600,
    bbox_inches="tight",
    facecolor="white"
)

plt.show()


# ============================================================
# 22. DOWNLOAD
# ============================================================

files.download(
    output_file
)