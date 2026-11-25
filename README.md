# English ASR Benchmark

Reproducible benchmark of three open-source English speech-to-text models on short-form, conversational, and long-form audio — measuring accuracy, throughput, and GPU memory on a single consumer GPU.

Built to answer a practical question: **which open ASR model should you deploy for transcribing recorded phone calls?**

---

## TL;DR

| Model | WER (short clips) | WER (20-min audio) | Throughput | Peak VRAM |
|---|---|---|---|---|
| **Parakeet-TDT-0.6B-v2** | 7.8% | **35.5%** | **172×** | 5.5 GB |
| Canary-1B-Flash | **7.1%** | 46.3% | 42× | 7.4 GB |
| faster-whisper-large-v3 | 7.3% | 43.8% | 10× | 4.7 GB |

*Up next: Qwen3-ASR-1.7B, FireRedASR2-AED (English support pending confirmation) — see [`models.md`](models.md).*

*Measured on an NVIDIA T4. Lower WER is better; higher throughput is better.*

**Finding:** on clean 15-second clips all three models are within one point of each other — statistically tied. On 20-minute recordings they separate by 8–11 points. **Short-clip benchmarks do not predict long-form performance.** If your production audio is long, benchmark on long audio.

**Recommendation:** Parakeet-TDT-0.6B-v2 — equal accuracy on clean audio, materially better on long audio, 4–17× the throughput, and memory that doesn't grow with recording length.

---

## Why this exists

Published ASR benchmarks are dominated by clean, short, read-aloud speech (LibriSpeech and friends). Production call audio is none of those things: it's long, conversational, narrowband, and often overlapping. A model that tops a leaderboard can be unusable on a 20-minute call.

This repo tests the same models across three deliberately different data tiers to expose where they break.

## Method

### Three data tiers

| Tier | Data | Size | Question it answers |
|---|---|---|---|
| **A** | [People's Speech](https://mlcommons.org/datasets/peoples-speech/) — 257 clips, 15–20 s | ~60 min, ~12k words | Primary accuracy number |
| **B** | Two-party phone conversations, references hand-verified | ~7 min, ~930 words | Does it hold up on real conversation? |
| **C** | [AMI Corpus](https://groups.inf.ed.ac.uk/ami/corpus/) far-field, 3 files of 17–26 min, each also band-limited to 8 kHz | ~63 min × 2 | Long-form behaviour, speed, VRAM, narrowband penalty |

Tier A alone would have concluded all three models are equivalent. Tier C showed that conclusion is wrong.

### Metrics

- **WER** — corpus-level (micro-averaged): `(ΣS + ΣD + ΣI) / ΣN`, summed across files and divided once. Not the mean of per-file WERs, which would weight a 12-word clip like a 500-word one.
- **S / D / I** — substitutions, deletions, insertions recorded separately. They mean different things downstream: deletions hurt keyword search, insertions hurt summarisation.
- **RTFx** — audio seconds transcribed per wall-clock second.
- **Peak VRAM** — determines which GPU you need.

Both reference and hypothesis are normalised with Whisper's `EnglishTextNormalizer` before scoring, so models aren't penalised for formatting differences (`$25` vs `twenty five dollars`).

### Narrowband condition

Telephone audio carries only up to ~4 kHz. To measure what that costs, Tier C files were downsampled to 8 kHz and back to 16 kHz:

```bash
sox input.wav -r 8000 /tmp/tmp.wav
sox /tmp/tmp.wav -r 16000 output_8k.wav
```

## Running it

Open `asr_evaluation.ipynb` in Google Colab with a **T4 GPU** runtime.

```
Cell 1  install deps  →  Runtime > Restart session
Cell 2  mount Drive
Cell 3  config: set BASE_DIR, SHORT_AUDIO_DIR, SHORT_TEXT_DIR
Cell 4  verify every file resolves before spending GPU time
Cell 5-7  run benchmark  (~60-90 min for 3 models)
Cell 8  aggregate tables
Cell 9  figures (300 DPI)
Cell 10 inspect transcripts — do this before trusting any number
```

Expected directory layout:

```
asr_benchmark/
├── short_clips/           # Tier A: audio + matching .txt
├── s1.wav, s1.txt         # Tier B
├── s5.wav, s5.txt
├── *.Array1-01.wav        # Tier C, 16 kHz
├── *.Array1-01_8k.wav     # Tier C, band-limited
├── *_reference.txt
└── results/               # written by the notebook
```

### Adding a model

Subclass and register — the harness handles chunking, timing, and scoring:

```python
class MyModel:
    key, label = "mymodel", "My-Model-v1"
    def load(self): ...
    def batch(self, arrays): ...   # list of np.float32 → list of str
    def long(self, audio): ...     # single long array → str

REGISTRY["mymodel"] = MyModel
```

## Known limitations

Stated up front rather than buried:

- **No dataset here is genuine call-centre telephony.** All three tiers are proxies.
- **Tier C absolute WER is inflated and not comparable to published AMI results.** The AMI reference transcripts merge multiple speakers by timestamp, interleaving them mid-sentence — a perfect transcript would still score badly. Overlapping speech on a far-field mic accounts for a further large share of deletions. Tier C is used for *relative* ranking, speed, and the bandwidth comparison only.
- **Tier B is ~930 words** — a sanity check, not enough to rank models.
- **Canary's long-form result is partly a harness artefact.** Its ~40 s input cap forced 30 s chunking; the comparison isn't perfectly like-for-like.
- **VRAM measured two ways.** PyTorch allocator for NeMo models; `nvidia-smi` sampling for faster-whisper, whose CTranslate2 allocations are invisible to PyTorch. Not perfectly comparable.
- **Speed measured at batch size 1 without VAD**, on a T4. Production throughput will be higher.
- **No significance testing.** With ~±0.6 points of uncertainty on Tier A, the differences between models there are not established as real. Bootstrap CIs (Bisani & Ney 2004) and MAPSSWE (Gillick & Cox 1989) are the right next step.

## Models

Three done, two more planned. See [`models.md`](models.md) for status and how to add one.

## Repo contents

```
common.py                  shared harness: read_audio, chunk, score() — import this, don't copy it
asr_evaluation.ipynb       Parakeet / Canary / faster-whisper (done)
oruk_fourier_2b_eval.ipynb Oruk-Fourier-2B (turnkey common.py notebook; H200 ref run complete, ready for T4 Drive run)
qwen3_asr_eval.ipynb       Qwen3-ASR-1.7B (planned)
fireredasr2_eval.ipynb     FireRedASR2-AED (planned — confirm English support first)
models.md                  status table + how to add a model
results/
  per_item.csv             per-file S/D/I/N, timing, VRAM — all models appended here
  summary_by_tier.csv      corpus WER per model per tier
  summary_by_band.csv      16 kHz vs 8 kHz
  figures/*.png            300 DPI, regenerated from per_item.csv
report/                    full write-up (current as of 3-model comparison)
```

New models append rows to `results/per_item.csv` rather than creating parallel files, so every figure and table always reflects every model tested so far without hand-merging anything.

**Audio data is not committed** — AMI and People's Speech should be obtained from their original sources under their own licences.

## References

**Models** — [Parakeet-TDT-0.6B-v2](https://huggingface.co/nvidia/parakeet-tdt-0.6b-v2) · [Canary-1B-Flash](https://huggingface.co/nvidia/canary-1b-flash) · [faster-whisper](https://github.com/SYSTRAN/faster-whisper) · [NeMo](https://github.com/NVIDIA/NeMo)

**Datasets** — Galvez et al., *The People's Speech*, NeurIPS D&B 2021 (CC-BY) · Carletta et al., *The AMI Meeting Corpus*, 2005

**Methodology**
- Radford et al., *Robust Speech Recognition via Large-Scale Weak Supervision*, 2022 — source of the English text normalizer
- Gandhi, von Platen & Rush, *ESB: A Benchmark for Multi-Domain End-to-End Speech Recognition*, 2022 — rationale for normalisation
- NIST [SCTK/sclite](https://github.com/usnistgov/SCTK) — reference implementation of corpus-level WER
- Bisani & Ney, *Bootstrap Estimates for Confidence Intervals in ASR Performance Evaluation*, ICASSP 2004
- Gillick & Cox, *Some Statistical Issues in the Comparison of Speech Recognition Algorithms*, ICASSP 1989
- [Open ASR Leaderboard](https://huggingface.co/spaces/hf-audio/open_asr_leaderboard)

## Licence

Code: MIT. Datasets retain their original licences and are not redistributed here.
