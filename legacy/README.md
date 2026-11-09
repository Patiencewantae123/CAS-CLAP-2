# Legacy code: not used for any reported result

Kept for history only. **Nothing in this folder may be cited as an experimental result.**

| Item | Why it is here |
|---|---|
| `generate_paper_results.py` | Draws "paper figures" from hard-coded benchmark numbers (FAD/CLAP for IM2WAV, AudioLDM-2, ...) and randomly generated curves. No model is evaluated. |
| `generate_advanced_audio.py` | Untrained, randomly initialised networks; prints FAD/SNR/CLAP drawn from `np.random.uniform`. |
| `paper_results/` | Output of the scripts above. Synthetic. |
| `MSA-I2A/` | Architecture skeleton. Its dataset returns `torch.randn`, inference ignores the input image, validation is a stub. Never trained. |
| `generated_test_input.jpg`, `output_sound.wav` | Artifacts of `generate_advanced_audio.py`. |
| `app_prototype.py` | The former `app.py` (until 2026-11-09). Scene, mood and "conflict" scores come from pixel brightness and colour, candidates from a hand-written list of 10 sounds, and audio is sine tones and noise. Not the CAS-V method; the current `app.py` runs the real pipeline. |
| `app_heuristic.py` | Earlier, longer version of the same heuristic prototype. |

All reported results come from the CAS-V pipeline (`casv/`, `scripts/`) and are
collected in `journal_results/`.
