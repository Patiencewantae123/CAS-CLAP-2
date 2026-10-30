<img width="1536" height="1024" alt="casv architecture" src="https://github.com/user-attachments/assets/a20d2039-bedb-4b4a-a488-2e2b08a4ebe2" />

# CAS-V: Conflict-Aware Semantic Verification for Image-to-Audio Generation

CAS-V recognises the scene and objects in an image, retrieves candidate sound
descriptions, **verifies** each candidate against the visual evidence (rejecting
indoor/outdoor contradictions and sound sources that are not in the image), and
prompts a text-to-audio generator with the verified description.

Results for the paper are collected in [`journal_results/`](journal_results/RESULTS.md).

## Repository layout

```
casv/                 CAS-V library
  vocab.py              Places365 / ImageNet / AudioCaps vocabularies, sound-source lexicon
  verifier.py           CLIP recognition, candidate retrieval, verification (measure + decide)
  audio.py              AudioLDM generation, mel spectrograms, CLAP scoring
scripts/              entry points (run from the repo root with  py -m scripts.<name>)
  fetch_audiocaps.py    build image + ground-truth audio pairs from AudioCaps
  fetch_category_examples.py  one test clip per figure category (fixed caption rule)
  eval_places365.py     scene / indoor-outdoor accuracy on Places365 val
  tune_casv.py          tune the verifier on AudioCaps val, score once on test
  run_casv_experiment.py  generate baseline vs CAS-V audio and score with CLAP
  make_casv_graphs.py   result graphs
  make_casv_figure.py   qualitative comparison figures
  build_journal_results.py  collect figures, tables and RESULTS.md
data/                 datasets (see data/README.md); downloaded media is git-ignored
eval/                 small smoke-test set
results/              raw outputs of every run
journal_results/      graphs/ (charts), qualitative/ (result images), tables/ (CSV + LaTeX), RESULTS.md
docs/                 target figure layout
notebooks/            exploratory Colab notebooks
legacy/               earlier code; synthetic outputs, not results (see legacy/README.md)
app.py                Streamlit demo (runtime files in app_data/)
```

## Reproducing the results

```
py -m pip install torch transformers diffusers accelerate soundfile scipy matplotlib yt-dlp[default] imageio-ffmpeg

# data (Places365 val ~500 MB; AudioCaps clips via yt-dlp, needs Node.js)
py -m scripts.fetch_audiocaps --split test --n 100
py -m scripts.fetch_audiocaps --split val --n 100
py -m scripts.fetch_category_examples

# experiments
py -m scripts.eval_places365 --per-class 10
py -m scripts.tune_casv --val data/audiocaps/pairs/val/manifest.json --test data/audiocaps/pairs/test/manifest.json
py -m scripts.run_casv_experiment --manifest data/audiocaps/pairs/test/manifest.json --run-name audiocaps_test100
py -m scripts.run_casv_experiment --manifest data/audiocaps/pairs/categories/manifest.json --run-name categories7

# figures and journal package
py -m scripts.make_casv_graphs --run results/audiocaps_test100
py -m scripts.make_casv_figure --run results/audiocaps_test100 --count 20
py -m scripts.make_casv_figure --run results/categories7 --count 7
py -m scripts.build_journal_results --run results/audiocaps_test100
```

See [`data/README.md`](data/README.md) for every dataset: source, licence, size and layout.

Places365 val is downloaded from `http://data.csail.mit.edu/places/places365/`
(`val_256.tar` into `data/places365/`, `filelist_places365-standard.tar` into `data/places365/filelists/`).

## Evaluation protocol

- Candidate captions come only from the AudioCaps **train** split.
- Verifier settings are tuned on AudioCaps **val**; the test split is evaluated once.
- Baseline and CAS-V use the same generator and seed; only the prompt differs.
- Every mean is reported with a bootstrap 95% CI and n; paired differences use Wilcoxon signed-rank.
