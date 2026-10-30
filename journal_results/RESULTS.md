# CAS-V results

Generated 2026-10-30 by `scripts/build_journal_results.py`. Every number below is read
from result files in `results/`; nothing is entered by hand.

## Setup

| Component | Choice |
|---|---|
| Recognition | CLIP ViT-B/32 zero-shot: Places365 scenes (365), ImageNet objects (1000) |
| Sound candidates | AudioCaps **train** captions (test/val captions never enter the candidate pool) |
| Verification | visual + scene + object support; explicit conflicts: indoor/outdoor, absent sound source |
| Generator | AudioLDM-S (`cvssp/audioldm-s-full-v2`), same seed for both arms |
| Evaluation | AudioCaps test clips: video frame as input, real recording + 5 human captions as ground truth; CLAP (`laion/clap-htsat-unfused`) |
| Baseline | same generator prompted with the top-1 retrieved caption, unverified |
| Tuning | grid search on AudioCaps **val**; test scored once |

## Figures

| File | Content |
|---|---|
| `figures/fig1_scene_recognition.png` | Places365 accuracy and confidence calibration |
| `figures/fig2_prompt_selection.png` | Prompt selection on val/test with 95% CIs |
| `figures/fig3_generated_audio_clap.png` | Generated audio: baseline vs CAS-V, paired |
| `figures/fig4_verification_statistics.png` | Candidate decisions and conflict types |
| `figures/fig5*_qualitative_examples.png` | First 20 test items in sample order (not selected) |

LaTeX versions of all tables are in `tables/`.

## Table 1. Scene recognition on Places365 val

| Metric | Value | n |
|---|---|---|
| Scene top-1 accuracy (365-way) | 38.8% | 3650 |
| Scene top-5 accuracy | 68.5% | 3650 |
| Indoor/outdoor accuracy | 91.6% | 3650 |
| Indoor/outdoor accuracy, p >= 0.7 | 95.5% | 3194 |

## Limitations

- Sample sizes are small (100 test pairs); report confidence intervals with every mean.
- CLAP similarity is an automatic proxy; no listening test has been run.
- AudioCaps frames are single video frames; off-screen sounds (narration, music) in the ground truth
  cannot be recovered from the image by any method.
- Places365/ImageNet are used for zero-shot vocabularies and Places365 val for recognition accuracy only;
  they contain no audio.

## Missing inputs at build time

- `results\tuning\fig_prompt_selection.png`
- `results\audiocaps_test100\fig_clap_comparison.png`
- `results\audiocaps_test100\fig_verification_stats.png`
- `results/tuning/tuning_results.json`
- `results\audiocaps_test100\results.json`
- `results\audiocaps_test100\graph_stats.json`
