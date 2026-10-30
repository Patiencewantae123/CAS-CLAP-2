# CAS-V results

Generated 2026-10-30 by `scripts/report.py`. Every number below is read
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
| `graphs/fig1_scene_recognition.png` | Places365 accuracy and confidence calibration |
| `graphs/fig2_prompt_selection.png` | Prompt selection on val/test with 95% CIs |
| `graphs/fig3_generated_audio_clap.png` | Generated audio: baseline vs CAS-V, paired |
| `graphs/fig4_verification_statistics.png` | Candidate decisions and conflict types |
| `qualitative/fig5_categories.png` | One AudioCaps test clip per category: first clip in seeded order whose captions match the category rule and whose frame passed visual screening (frames only, before generation). Rejections and the one relabel (forest → field) are listed in `data/audiocaps/pairs/categories/selection_rule.json` |
| `qualitative/figS1*_random20.png` | Supplementary: first 20 items of the random test sample (not selected) |

LaTeX versions of all tables are in `tables/`.

## Table 1. Scene recognition on Places365 val

| Metric | Value | n |
|---|---|---|
| Scene top-1 accuracy (365-way) | 38.8% | 3650 |
| Scene top-5 accuracy | 68.5% | 3650 |
| Indoor/outdoor accuracy | 91.6% | 3650 |
| Indoor/outdoor accuracy, p >= 0.7 | 95.5% | 3194 |

## Table 2. Prompt selection: CLAP(prompt, ground-truth recording)

| Split | Method | Mean [95% CI] | n |
|---|---|---|---|
| val | baseline | 0.306 [0.265, 0.346] | 100 |
| val | default | 0.307 [0.266, 0.348] | 100 |
| val | tuned | 0.338 [0.298, 0.377] | 100 |
| val | oracle | 0.513 [0.480, 0.545] | 100 |
| test | baseline | 0.265 [0.220, 0.309] | 100 |
| test | default | 0.290 [0.248, 0.333] | 100 |
| test | tuned | 0.278 [0.235, 0.322] | 100 |
| test | oracle | 0.509 [0.479, 0.538] | 100 |

- Test, default − baseline: 0.025 [-0.009, 0.058]; better on 34, worse on 27, unchanged on 39 items.
- Test, tuned − baseline: 0.013 [-0.016, 0.040]; better on 25, worse on 20, unchanged on 55 items.
- Tuned configuration: `{"retrieve_k": 20, "support_top_n": 5, "w_visual": 0.5, "w_scene": 0.0, "w_object": 0.5, "accept_threshold": 0.5, "repair_threshold": 0.4, "env_conflict_confidence": 1.01, "source_absent_z": -99.0, "uncertainty_threshold": 0.1}`
- The primary CAS-V configuration is the default, fixed before any data was seen; the tuned configuration is an ablation.
- The val-optimal configuration disables both explicit conflict checks: under this objective, rejecting sounds that are not visible does not increase agreement with the recording, which itself contains off-screen sounds (narration, music, speech).
- Selection headroom on test: oracle − baseline = +0.244.

## Table 5. Figure 5 examples: CLAP(generated audio, GT recording)

| Category | AudioCaps id | Prompt changed | Baseline | CAS-V | Δ |
|---|---|---|---|---|---|
| busy road | kgjNIDmO8a8 | yes | 0.287 | 0.275 | -0.012 |
| office | JQz40TkjymY | yes | 0.489 | 0.543 | +0.054 |
| beach | 3MoF8myFs8Y | yes | 0.506 | 0.340 | -0.165 |
| field (wind / birds) | 0yxEvdnimGg | yes | 0.162 | 0.112 | -0.050 |
| cloth scratch | zwoqJY03yHE | no | 0.285 | 0.285 | +0.000 |
| leaves | fmEft49sPfE | yes | 0.284 | 0.257 | -0.027 |
| metal hit | LB6CZ0x-kns | yes | 0.544 | 0.495 | -0.050 |
| Mean |  | 6/7 | 0.365 | 0.330 | -0.036 |

## Table 3. Generated audio (AudioLDM-S), baseline vs CAS-V

| Metric | Baseline | CAS-V | Δ [95% CI] | better/worse | Wilcoxon p | n |
|---|---|---|---|---|---|---|
| CLAP(audio, GT captions) | 0.159 | 0.162 | +0.003 [-0.028, +0.033] | 29/32 | 0.923 | 100 |
| CLAP(audio, GT recording) | 0.307 | 0.325 | +0.018 [-0.014, +0.051] | 33/28 | 0.277 | 100 |

## Table 4. Verification statistics

| Quantity | Count |
|---|---|
| Candidates ACCEPT | 436 |
| Candidates REPAIR | 774 |
| Candidates REJECT | 790 |
| Images ACCEPT | 91 |
| Images REPAIR | 2 |
| Images UNCERTAIN | 7 |
| Conflict: absent source: a motor vehicle with an engine | 74 |
| Conflict: environment: outdoor sound | 72 |
| Conflict: absent source: cars on a road | 58 |
| Conflict: absent source: insects | 7 |
| Conflict: absent source: an emergency vehicle | 6 |
| Conflict: absent source: a clock | 6 |
| Conflict: absent source: a stream or river | 4 |
| Conflict: absent source: a boat | 3 |

## Limitations

- Sample sizes are small (100 test pairs); report confidence intervals with every mean.
- CLAP similarity is an automatic proxy; no listening test has been run.
- AudioCaps frames are single video frames; off-screen sounds (narration, music) in the ground truth
  cannot be recovered from the image by any method.
- Places365/ImageNet are used for zero-shot vocabularies and Places365 val for recognition accuracy only;
  they contain no audio.
