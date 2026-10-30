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
  stats.py              bootstrap / Wilson confidence intervals, paired differences
  log.py                logging setup shared by all scripts
scripts/              entry points (run from the repo root with  py -m scripts.<name>)
  run_all.py       runs every stage below in order
  download_data.py    build image + ground-truth audio pairs from AudioCaps
  select_examples.py  one test clip per figure category (fixed caption rule)
  eval_scenes.py     scene / indoor-outdoor accuracy on Places365 val
  train.py          tune the verifier on AudioCaps val, score once on test
  test.py  generate baseline vs CAS-V audio and score with CLAP
  plots.py   result graphs
  figures.py   qualitative comparison figures
  listening_test.py     blind A/B listening test: build the rater package, analyse ratings
  report.py  collect figures, tables and RESULTS.md
data/                 datasets (see data/README.md); downloaded media is git-ignored
eval/                 small smoke-test set
tests/                pytest suite (verifier logic, vocabularies, statistics, figure text)
results/              raw outputs of every run
journal_results/      graphs/ (charts), comparisons/<set>/ (comparison images), tables/ (CSV + LaTeX), RESULTS.md
docs/                 target figure layout
notebooks/            exploratory Colab notebooks
legacy/               earlier code; synthetic outputs, not results (see legacy/README.md)
demo_app.py           Streamlit demo of the real CAS-V pipeline (streamlit run demo_app.py)
app.py                earlier heuristic prototype (pixel statistics, sine tones); not the paper's method
evaluation/            blind listening-test package for raters
```

## Reproducing the results

```
py -m pip install -r requirements.txt     # exact versions used for journal_results/
py -m scripts.run_all                # everything below, in order (hours on CPU)
py -m scripts.run_all --list         # show the stages
py -m scripts.run_all --from figures # only re-render figures and the journal package
```

Individual stages:

```
# data (Places365 val ~500 MB; AudioCaps clips via yt-dlp, needs Node.js)
py -m scripts.download_data --split test --n 100
py -m scripts.download_data --split test --n 400 --out data/audiocaps/pairs/test400
py -m scripts.download_data --split val --n 100
py -m scripts.select_examples --screening data/audiocaps/pairs/categories/screening/screening.json

# experiments
py -m scripts.eval_scenes --per-class 10
py -m scripts.train --val data/audiocaps/pairs/val/manifest.json --test data/audiocaps/pairs/test/manifest.json
py -m scripts.train --val data/audiocaps/pairs/val/manifest.json --test data/audiocaps/pairs/test400/manifest.json --out results/tuning_test400
py -m scripts.test --manifest data/audiocaps/pairs/test/manifest.json --run-name audiocaps_test100
py -m scripts.test --manifest data/audiocaps/pairs/test400/manifest.json --run-name audiocaps_test400 --reuse-from results/audiocaps_test100
py -m scripts.test --manifest data/audiocaps/pairs/categories/manifest.json --run-name categories7

# figures and journal package
py -m scripts.plots --run results/audiocaps_test400 --tuning results/tuning_test400/tuning_results.json --earlier-run results/audiocaps_test100
py -m scripts.figures --run results/audiocaps_test400 --count 20
py -m scripts.figures --run results/categories7 --count 7
py -m scripts.report
```

See [`data/README.md`](data/README.md) for every dataset: source, licence, size and layout.

Places365 val is downloaded from `http://data.csail.mit.edu/places/places365/`
(`val_256.tar` into `data/places365/`, `filelist_places365-standard.tar` into `data/places365/filelists/`).

## Evaluation protocol

- Candidate captions come only from the AudioCaps **train** split.
- Verifier settings are tuned on AudioCaps **val** and fixed before test is scored; test is never used for tuning.
- The test sample was extended from 100 to 400 clips in the same seeded order, with no change to the method.
- Baseline and CAS-V use the same generator and seed; only the prompt differs.
- Every mean is reported with a bootstrap 95% CI and n; paired differences use Wilcoxon signed-rank.

## Development

```
py -m pytest                         # 25 tests, ~10 s, no model downloads
py -m ruff check casv scripts tests  # lint
py -m ruff format casv scripts tests # format
```

## Demo

```
streamlit run demo_app.py
```

Upload an image or pick a Figure 5 example: recognition, all 20 candidates with
their verification scores, and the baseline vs CAS-V prompts appear instantly.
Audio is generated on request (about 1-2 min per clip on CPU) and saved to
`app_data/runs/`.

## Listening test

```
py -m scripts.listening_test build --run results/audiocaps_test100   # evaluation/listening_test/
py -m scripts.listening_test analyze --ratings "evaluation/ratings/*.csv"
```

Send `evaluation/listening_test/` (open `index.html`) to 2-3 raters. Items are the
test clips where CAS-V changed the prompt; A/B order is randomised and the key is kept in
`results/listening_test/key.json`, so do not share the rest of the repository with raters.
