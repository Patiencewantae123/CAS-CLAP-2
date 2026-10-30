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
| Evaluation | AudioCaps test clips: video frame as input, real recording + 5 human captions as ground truth; CLAP (`laion/clap-htsat-unfused`), clips over 10 s cropped to their first 10 s |
| Baseline | same generator prompted with the top-1 retrieved caption, unverified |
| Tuning | grid search on AudioCaps **val**; test is never used for tuning |

## Figures

| File | Content |
|---|---|
| `graphs/fig1_scene_recognition.png` | Places365 accuracy and confidence calibration |
| `graphs/fig2_prompt_selection.png` | Prompt selection on val/test with 95% CIs |
| `graphs/fig3_generated_audio_clap.png` | Generated audio: baseline vs CAS-V, paired |
| `graphs/fig4_verification_statistics.png` | Candidate decisions and conflict types |
| `comparisons/fig5_categories/` | One AudioCaps test clip per category: first clip in seeded order whose captions match the category rule and whose frame passed visual screening (frames only, before generation). Rejections and the one relabel (forest → field) are listed in `data/audiocaps/pairs/categories/selection_rule.json` |
| `comparisons/figS1_random20/` | Supplementary: first 20 items of the random test sample (not selected) |

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
| val | baseline | 0.308 [0.268, 0.348] | 100 |
| val | default | 0.309 [0.268, 0.350] | 100 |
| val | tuned | 0.340 [0.301, 0.380] | 100 |
| val | oracle | 0.515 [0.482, 0.547] | 100 |
| test | baseline | 0.287 [0.265, 0.309] | 400 |
| test | default | 0.289 [0.268, 0.311] | 400 |
| test | tuned | 0.291 [0.269, 0.313] | 400 |
| test | oracle | 0.508 [0.492, 0.524] | 400 |

- Test, default − baseline: 0.003 [-0.013, 0.018]; better on 122, worse on 110, unchanged on 168 items.
- Test, tuned − baseline: 0.005 [-0.008, 0.017]; better on 92, worse on 68, unchanged on 240 items.
- Tuned configuration: `{"retrieve_k": 20, "support_top_n": 5, "w_visual": 0.5, "w_scene": 0.0, "w_object": 0.5, "accept_threshold": 0.5, "repair_threshold": 0.4, "env_conflict_confidence": 1.01, "source_absent_z": -99.0, "uncertainty_threshold": 0.1}`
- The primary CAS-V configuration is the default, fixed before any data was seen; the tuned configuration is an ablation.
- The val-optimal configuration disables both explicit conflict checks: under this objective, rejecting sounds that are not visible does not increase agreement with the recording, which itself contains off-screen sounds (narration, music, speech).
- Selection headroom on test: oracle − baseline = +0.222.

## Table 5. Figure 5 examples: CLAP(generated audio, GT recording)

| Category | AudioCaps id | Prompt changed | Baseline | CAS-V | Δ |
|---|---|---|---|---|---|
| busy road | kgjNIDmO8a8 | yes | 0.279 | 0.261 | -0.018 |
| office | JQz40TkjymY | yes | 0.481 | 0.535 | +0.053 |
| beach | 3MoF8myFs8Y | yes | 0.535 | 0.359 | -0.176 |
| field (wind / birds) | 0yxEvdnimGg | yes | 0.155 | 0.111 | -0.043 |
| cloth scratch | zwoqJY03yHE | no | 0.292 | 0.292 | +0.000 |
| leaves | fmEft49sPfE | yes | 0.275 | 0.257 | -0.018 |
| metal hit | LB6CZ0x-kns | yes | 0.526 | 0.495 | -0.031 |
| Mean |  | 6/7 | 0.363 | 0.330 | -0.033 |

## Table 3. Generated audio (AudioLDM-S), baseline vs CAS-V

| Metric | Baseline | CAS-V | Δ [95% CI] | better/worse | Wilcoxon p | n |
|---|---|---|---|---|---|---|
| CLAP(audio, GT captions) | 0.183 | 0.186 | +0.003 [-0.013, +0.018] | 120/112 | 0.714 | 400 |
| CLAP(audio, GT recording) | 0.340 | 0.343 | +0.003 [-0.012, +0.019] | 121/111 | 0.728 | 400 |
| CLAP(audio, GT captions), new items only | 0.189 | 0.193 | +0.003 [-0.015, +0.021] | 90/78 | 0.696 | 297 |
| CLAP(audio, GT recording), new items only | 0.349 | 0.348 | -0.001 [-0.019, +0.017] | 88/80 | 0.846 | 297 |

- The primary result uses all 400 items, drawn in the same seeded clip order as the earlier test results; an earlier clip is absent here if its download failed. The new-items rows are a sensitivity analysis on the 297 clips in no earlier run (audiocaps_test100: 98 of its 100 items are in this sample, same verifier configuration; categories7: 6 of its 7 items are in this sample, same verifier configuration).

## Table 4. Verification statistics

| Quantity | Count |
|---|---|
| Candidates ACCEPT | 1731 |
| Candidates REPAIR | 3137 |
| Candidates REJECT | 3132 |
| Images ACCEPT | 364 |
| Images REPAIR | 14 |
| Images UNCERTAIN | 22 |
| Conflict: environment: outdoor sound | 240 |
| Conflict: absent source: cars on a road | 214 |
| Conflict: absent source: a motor vehicle with an engine | 201 |
| Conflict: absent source: insects | 33 |
| Conflict: absent source: an emergency vehicle | 27 |
| Conflict: absent source: a clock | 16 |
| Conflict: absent source: a cat | 15 |
| Conflict: absent source: food frying in a pan | 12 |

## Limitations

- Results rest on 400 test pairs; report confidence intervals with every mean.
- CLAP similarity is an automatic proxy; no listening test has been run.
- CLAP embeddings of a recording change when it is shifted by a few milliseconds, so single-item scores
  are noisy; compare means with their confidence intervals, not individual items.
- AudioCaps frames are single video frames; off-screen sounds (narration, music) in the ground truth
  cannot be recovered from the image by any method.
- Places365/ImageNet are used for zero-shot vocabularies and Places365 val for recognition accuracy only;
  they contain no audio.
