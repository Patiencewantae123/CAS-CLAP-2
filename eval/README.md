# CAS-V evaluation

## 1. Prepare image + ground-truth pairs

Each item needs an image and, for the bottom row of the figure and the
audio-to-audio metric, the **real recording** that belongs to that image.
Good sources:

| Category | Dataset |
|---|---|
| busy road, office, beach, forest | VGGSound, AudioSet (video frame + its audio track) |
| cloth scratch, leaves, metal hit | Greatest Hits (Owens et al., "Visually Indicated Sounds") |

Take one frame from the clip as the image and the clip's audio (WAV/FLAC, any
sample rate, it is resampled to 16 kHz) as `gt_audio`.

```
eval/
  images/busy_road.jpg
  audio/busy_road.wav
  manifest.json
```

`manifest.json` (paths are relative to the manifest):

```json
[
  {"id": "busy_road", "image": "images/busy_road.jpg", "gt_audio": "audio/busy_road.wav", "label": "busy road"},
  {"id": "office",    "image": "images/office.jpg",    "gt_audio": "audio/office.wav",    "label": "office"}
]
```

`label` must be a scene key from `casv/concepts.py` (`busy road`, `office`,
`beach`, `forest`, `cloth scratch`, `leaves`, `metal hit`, ...). It is used
only for evaluation, never as input to CAS-V.

## 2. Run

```
py run_casv_experiment.py --manifest eval/manifest.json --run-name main
py make_casv_figure.py --run results/main --items busy_road,office,beach,forest,cloth_scratch,leaves,metal_hit
```

Use the default `--steps 50` for reported results. On CPU this takes a few
minutes per item; on a GPU it is seconds.

## 3. What is measured

| Metric | Meaning |
|---|---|
| `scene_accuracy` | CLIP top-1 scene matches `label` |
| `clap_text_{baseline,casv}` | CLAP cosine between generated audio and the label's caption text |
| `clap_gt_audio_{baseline,casv}` | CLAP cosine between generated audio and the ground-truth recording |

The baseline uses the same generator and seed, prompted with the unverified
top-k CLIP concepts, so any difference comes from verification alone.
Report the number of items (`n`) alongside every mean; with a handful of
images the differences are anecdotal, not a benchmark result.
