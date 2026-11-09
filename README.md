<img width="1536" height="1024" alt="casv architecture" src="https://github.com/user-attachments/assets/a20d2039-bedb-4b4a-a488-2e2b08a4ebe2" />

# CAS-V: Conflict-Aware Semantic Verification for Image-to-Audio Generation

[![tests](https://github.com/Patiencewantae123/CAS-CLAP-2/actions/workflows/tests.yml/badge.svg)](https://github.com/Patiencewantae123/CAS-CLAP-2/actions/workflows/tests.yml)

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
  train.py          tune the verifier on AudioCaps val, score on test
  test.py  generate baseline vs CAS-V audio and score with CLAP
  rescore.py  recompute a finished run's CLAP metrics from its saved audio
  plots.py   result graphs
  figures.py   qualitative comparison figures
  select_figure_clips.py  rule-based choice of clear examples for the selected pages
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
app.py                Streamlit comparison board of the real pipeline, laid out like docs/target_figure.png
app_gradio.py         the same board in Gradio, for a free Hugging Face Space (ZeroGPU)
board.py              helpers shared by both boards (columns, clip cache, figure export)
space/                Hugging Face Space settings (README.md header, requirements.txt)
demo_app.py           single-image Streamlit demo of the same pipeline
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
py -m scripts.train --val data/audiocaps/pairs/val/manifest.json --test data/audiocaps/pairs/test400/manifest.json --out results/tuning_test400
py -m scripts.test --manifest data/audiocaps/pairs/test/manifest.json --run-name audiocaps_test100
py -m scripts.test --manifest data/audiocaps/pairs/test400/manifest.json --run-name audiocaps_test400 --reuse-from results/audiocaps_test100
py -m scripts.test --manifest data/audiocaps/pairs/categories/manifest.json --run-name categories7

# figures and journal package
py -m scripts.plots --run results/audiocaps_test400 --earlier-run results/audiocaps_test100 results/categories7
py -m scripts.figures --run results/audiocaps_test400 --count 20
py -m scripts.figures --run results/audiocaps_test400 --count 140 --folder comparisons_140   # 20 pages of 7 clips
py -m scripts.select_figure_clips --run results/audiocaps_test400                            # selected examples, rule in the JSON
py -m scripts.figures --run results/audiocaps_test400 --selection results/audiocaps_test400/selected_clips.json --folder comparisons_selected
py -m scripts.figures --run results/categories7 --count 7
py -m scripts.report
```

See [`data/README.md`](data/README.md) for every dataset: source, licence, size and layout.

Places365 val is downloaded from `http://data.csail.mit.edu/places/places365/`
(`val_256.tar` into `data/places365/`, `filelist_places365-standard.tar` into `data/places365/filelists/`).

## Evaluation protocol

- Candidate captions come only from the AudioCaps **train** split.
- Verifier settings are tuned on AudioCaps **val** and fixed before test is scored; test is never used for tuning.
- The 400-clip test sample uses the same seeded order as the earlier 100-clip sample; 98 clips are shared, and 2 earlier clips are absent because their downloads failed.
- Baseline and CAS-V use the same generator and seed; only the prompt differs.
- CLAP clips over 10 s are cropped to their first 10 s, so scores repeat exactly.
- Every mean is reported with a bootstrap 95% CI and n; paired differences use Wilcoxon signed-rank.

## Development

```
py -m pytest                         # 48 tests, ~15 s, no model downloads
py -m ruff check casv scripts tests app.py app_gradio.py board.py  # lint
py -m ruff format casv scripts tests app.py app_gradio.py board.py # format
```

## Demo

```
py -m streamlit run app.py        # comparison board (target-figure layout)
py -m streamlit run demo_app.py   # one image at a time
```

`app.py` puts one image per column, with the rows of the target figure: input
image, existing research (top-1 retrieved caption, unverified), ours (verified
concepts, generated audio, sound type) and the target sound. Columns are Figure 5
examples (with their real recordings) and/or your own uploads (up to 7 in all),
each with an optional real recording. Recognition, verified concepts and sound
types appear instantly. Audio is generated on request (about 20 s per clip at the
default settings on a recent CPU) with the same seed for both prompts, and cached by prompt and settings in
`app_data/runs/clips/`. A panel without real data says "not generated yet" or
"not provided". **Build figure** renders the board with `scripts/figures.py` into
`app_data/runs/figure_<timestamp>/`; its caption file says the images were picked
by hand, so such a figure is an illustration, not a result. A tab per image lists
the recognition and all 20 candidates with their verification scores, with a
"why this caption" graph: the candidates by score, coloured by decision, with the
thresholds, the baseline and the CAS-V choice, beside the recognised scenes and
objects (`board.verification_chart`). A **Paper results** section shows the
evaluation graphs from `journal_results/graphs/`.

`app_gradio.py` shows the board, a card per image to listen to all three sounds
(with the table of all candidates and their scores) and a summary table; the
paper's evaluation graphs sit under **More info** at the bottom.

## Hosting

**Permanent, free: Hugging Face Space on ZeroGPU.** Free ZeroGPU Spaces run Gradio
apps only, so the Space runs `app_gradio.py`: the same board, with recognition on
CPU and AudioLDM on a shared GPU charged to each visitor's daily ZeroGPU quota.
Uploads only; the Figure 5 recordings (YouTube audio) are not published.

1. Hugging Face account with a verified email, older than 30 days (needed for free ZeroGPU).
2. Create the Space at https://huggingface.co/new-space: SDK **Gradio**, hardware **ZeroGPU**, public.
3. Deploy it with one command:
   ```
   py -m scripts.deploy_space <user>/<space>
   ```
   The first time, it asks for a **Write** token from https://huggingface.co/settings/tokens
   (pasted into the terminal, hidden; it refuses incomplete or read-only tokens and saves a valid
   one). It checks the Space is yours and on ZeroGPU, builds the bundle (`scripts/build_space.py`:
   only the files the Space needs, never git-ignored ones, ~96 MB), uploads it in place of the
   Space's files, and follows the build until the Space runs, or prints the end of its error log.
4. The Space is then at `https://huggingface.co/spaces/<user>/<space>` (first build: 5-10 min).

Run the same command again after changing `app_gradio.py`, `board.py`, `casv/` or `space/`.
To try the Gradio app locally: `py -m pip install gradio==6.31.0 spaces` then `py app_gradio.py`.

**Temporary, from your PC.** `streamlit run app.py`, then in a second terminal
`cloudflared tunnel --url http://localhost:8501`; it prints a public
`https://….trycloudflare.com` link that works while both terminals run.

## Listening test

```
py -m scripts.listening_test build --seed <private number>   # evaluation/listening_test.zip
py -m scripts.listening_test analyze --ratings "evaluation/ratings/*.csv"
```

Send `evaluation/listening_test.zip` to at least 3 raters who do not know the project, and put the
returned CSVs in `evaluation/ratings/`. Items are the test clips where CAS-V changed the prompt.
A/B order is randomised, and this repository is public, so the seed and the answer key
(`results/listening_test/key.json`) stay out of git until rating has finished; commit the key then.

Analysis fixed before any ratings were collected. Primary: each item counts once, for the system
most of its raters preferred, with a two-sided sign test over items, ties excluded. Secondary: paired
per-item reduction in "contains a sound that does not belong" flags. Inter-rater agreement: Fleiss' kappa.
