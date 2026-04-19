# Speech Understanding - Programming Assignment 2

> **Robust Code-Switched Transcription and Zero-Shot Cross-Lingual Voice Cloning**

[![Python 3.10](https://img.shields.io/badge/python-3.10-blue.svg)](https://www.python.org/)
[![PyTorch 2.1+](https://img.shields.io/badge/pytorch-2.1%2B-red.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

---

## Overview

This repository implements a complete pipeline that:

1. **Denoises** a 10-minute code-switched (Hinglish) lecture segment using DeepFilterNet or Spectral Subtraction fallback.
2. **Identifies languages** at 200 ms frame resolution using a Wav2Vec2 multi-head classifier fine-tuned on FLEURS.
3. **Transcribes** the audio via Whisper with a custom N-gram logit biasing mechanism that prioritises speech-course technical terms.
4. **Converts** the Hinglish transcript to IPA using 40+ Hinglish phonology rules, then translates to Marathi using NLLB-200.
5. **Synthesises** the Marathi text with Meta MMS VITS and applies DTW + Praat PSOLA prosody warping to match the professor's speaking style.
6. **Evaluates adversarial robustness** with an LFCC-CNN anti-spoofing countermeasure (EER) and FGSM perturbation attack on the LID model.

**Target LRL:** Marathi (`mar_Deva`)  
**Source video:** [YouTube lecture](https://youtu.be/ZPUtA3W-7_I) — segment 2h20m → 2h30m

---

## Evaluation Benchmarks (Strict Passing Criteria)

| Metric                  | Target        | Description                                |
|-------------------------|---------------|--------------------------------------------|
| LID Macro F1            | ≥ 0.85        | Hindi / English frame classification       |
| WER — English segments  | < 15%         | Word Error Rate on English portions        |
| WER — Hindi segments    | < 25%         | Word Error Rate on Hindi portions          |
| MCD                     | < 8.0 dB      | Mel-Cepstral Distortion vs. reference      |
| Anti-Spoof EER          | < 10%         | Equal Error Rate of LFCC-CNN CM            |
| Switching Accuracy      | ≤ 200 ms      | Language boundary timestamp precision      |

---

## Repository Structure

```
Speech_ASS_2/
├── config.py               # Paths, models, hyperparameters
├── pipeline.py             # End-to-end orchestration (main entry point)
├── evaluate.py             # Standalone metrics evaluation
├── download_segment.py     # YouTube segment download helper
│
├── part1_denoise.py        # DeepFilterNet / Spectral Subtraction denoising
├── part1_lid.py            # Multi-head LID (Wav2Vec2 + FLEURS fine-tuning)
├── part1_stt.py            # Whisper + N-Gram logit biasing + WER
│
├── part2_translation.py    # Hinglish→IPA→Marathi translation pipeline
│
├── part3_tts_prosody.py    # MMS synthesis + DTW PSOLA prosody warping + MCD
│
├── part4_antispoof.py      # LFCC-CNN CM + EER + FGSM adversarial attack
│
├── data/
│   ├── original_segment.wav       ← YOU provide this (see Step 2)
│   ├── student_voice_ref.wav      ← YOU provide this (see Step 3)
│   ├── technical_corpus.json      # 100+ EN→HI→MAR term glossary
│   ├── syllabus.txt               # Auto-generated speech course terms
│   └── [outputs written here]
│
├── plots/                  # Spectrograms, DET curves, DTW paths, etc.
│
├── custom_lid_weights.pt   # Trained LID weights (after training)
├── cm_weights.pt           # Trained CM weights (after training)
│
├── report.tex              # IEEE two-column report (10 pages)
└── requirements.txt
```

---

## Environment Setup

### 1. Create and activate a Python 3.10 environment

```bash
conda create -n speech_ass2 python=3.10
conda activate speech_ass2
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Install system dependencies

| Dependency | Purpose                          | Install                                    |
|------------|----------------------------------|--------------------------------------------|
| `ffmpeg`   | Audio conversion for yt-dlp      | [ffmpeg.org](https://ffmpeg.org/download.html) → add to PATH |
| `espeak-ng`| IPA phonemizer backend           | `winget install eSpeak-NG.eSpeak-NG` (Windows) |

---

## Data Preparation

### Step 1 - Download the lecture segment (automated)

```bash
python download_segment.py
# Downloads 2h20m–2h30m from https://youtu.be/ZPUtA3W-7_I
# Output: data/original_segment.wav (16 kHz, mono, ~57 MB)
```

To use a different time window:

```bash
python download_segment.py --start 02:25:00 --end 02:35:00
```

### Step 2 - Record your reference voice

Place **your own 60-second voice recording** in the root directory named as:

```
Myvoice.m4a
```

Then run the preparation script to convert it to the required 16 kHz WAV format (this proves it's your own voice):

```bash
python prepare_student_voice.py
```

This will generate `data/student_voice_ref.wav` which the pipeline will use for zero-shot cloning.

---

## Running the Pipeline

### Full run (all stages)

```bash
python pipeline.py
```

### Skip stages (for resuming)

```bash
# Skip denoising (use existing data/denoised_segment.wav)
python pipeline.py --skip-denoise

# Skip LID fine-tuning (use existing custom_lid_weights.pt)
python pipeline.py --skip-train-lid

# Skip STT transcription (use existing data/transcript.txt)
python pipeline.py --skip-stt

# Skip translation (use existing data/translated.txt)
python pipeline.py --skip-translate

# Skip TTS synthesis
python pipeline.py --skip-tts

# Skip CM training
python pipeline.py --skip-train-cm
```

### Evaluate only (no training)

```bash
python pipeline.py --only-evaluate
# or
python evaluate.py
```

### Run individual modules

```bash
# Denoise
python part1_denoise.py

# LID: train, evaluate, or infer
python part1_lid.py --train
python part1_lid.py --evaluate
python part1_lid.py --infer

# STT
python part1_stt.py

# Translation
python part2_translation.py

# TTS
python part3_tts_prosody.py

# Anti-spoofing
python part4_antispoof.py --train
python part4_antispoof.py --evaluate
python part4_antispoof.py --attack
```

---

## Output Files

| File                             | Description                                        |
|----------------------------------|----------------------------------------------------|
| `data/denoised_segment.wav`      | DeepFilterNet or Spectral-Subtracted audio         |
| `data/lid_timestamps.csv`        | Per-frame language labels at 200 ms resolution     |
| `data/transcript.txt`            | Code-switched Hinglish transcript                  |
| `data/ipa_transcript.txt`        | IPA representation of the transcript               |
| `data/translated.txt`            | Marathi translation                                |
| `data/output_LRL_cloned.wav`     | Final prosody-warped synthesis at 22,050 Hz        |
| `data/metrics_report.json`       | All 5 evaluation metrics (JSON)                    |
| `plots/spectrogram_compare.png`  | Before/after denoising spectrogram                 |
| `plots/lid_confusion.png`        | LID confusion matrix                               |
| `plots/dtw_path.png`             | DTW alignment path                                 |
| `plots/f0_comparison.png`        | F0 contour before and after prosody warping        |
| `plots/det_curve.png`            | Anti-spoofing DET curve with EER marked            |
| `plots/fgsm_epsilon.png`         | FGSM epsilon sweep vs. misclassification rate      |

---

## Troubleshooting

| Issue | Solution |
|-------|----------|
| `espeak not found` | Install `espeak-ng` system-wide; IPA will fall back to basic rules otherwise |
| CUDA OOM on Whisper | Switch to `openai/whisper-small` in `config.py` |
| `deepfilternet` install fails | Use `pip install deepfilternet --no-build-isolation`; pipeline falls back to spectral subtraction |
| `yt-dlp` download fails | Check that `ffmpeg` is on PATH; try `yt-dlp --version` |
| DTW runs out of memory | Reduce audio to 5 minutes; MFCC matrices can be large |
| MMS TTS fails (OOM) | Lower chunk size in `generate_base_tts()` (change `80` words to `40`) |

---

## Mathematical Highlights

### N-Gram Logit Bias (Task 1.2)
$$\tilde{v}_i = v_i + B \cdot \mathbf{1}[x_i \in \mathrm{Prefix}(\mathcal{S})] + \beta B \cdot \mathbf{1}[x_i = \mathrm{first}(s), s \in \mathcal{S}]$$

### DTW F0 Mapping (Task 3.2)
$$F_0^{\mathrm{warped}}(t_j) = F_0^{\mathrm{src}}\bigl(t_{i^*(j)}\bigr), \quad i^*(j) = \arg\min_{(i,j')\in\mathcal{W}} |j' - j|$$

### FGSM Attack (Task 4.2)
$$x_{\mathrm{adv}} = x - \varepsilon \cdot \mathrm{sign}\!\left(\nabla_x \mathcal{L}(f_\theta(x), 0)\right)$$

---

## Report

The IEEE two-column report (`report.tex`) covers:
- Mathematical formulation of N-gram logit biasing
- Hinglish phonology rule table
- DTW + PSOLA prosody warping derivation
- Ablation study (flat vs. warped synthesis)
- Anti-spoofing architecture and EER analysis
- FGSM adversarial robustness results
- One non-obvious design choice per task

Compile with:
```bash
pdflatex report.tex
```

---

## References

- Radford et al., *Whisper*, arXiv:2212.04356, 2022
- Baevski et al., *wav2vec 2.0*, NeurIPS 2020
- Costa-jussà et al., *NLLB-200*, arXiv:2207.04672, 2022
- Pratap et al., *MMS*, arXiv:2305.13516, 2023
- Schröter et al., *DeepFilterNet*, ICASSP 2023
- Desplanques et al., *ECAPA-TDNN*, Interspeech 2020
- Goodfellow et al., *FGSM*, ICLR 2015
