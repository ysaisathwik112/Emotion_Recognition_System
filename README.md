# 🎙️ AI-Powered Speech Emotion Recognition Platform

> Enterprise-grade end-to-end Speech Emotion Recognition system combining deep learning, signal processing, and NLP — deployable on Streamlit Cloud.

![Python](https://img.shields.io/badge/Python-3.11%2B-3776AB?logo=python&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-1.32%2B-FF4B4B?logo=streamlit&logoColor=white)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.15%2B-FF6F00?logo=tensorflow&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-16A34A)

---

## Overview

This platform classifies human emotions from speech audio using a modular ML pipeline:

| Layer | Technology |
|---|---|
| Audio I/O | Librosa · SoundFile · PyDub |
| Feature Extraction | MFCC · Spectral · Chroma · Mel · Tonnetz · Pitch |
| Deep Learning | CNN · BiLSTM · CNN-LSTM · Wav2Vec2 |
| NLP | OpenAI Whisper STT · DistilRoBERTa text emotion |
| Explainability | SHAP · Gradient-based feature importance |
| Reporting | PDF · CSV · Excel |
| UI | Streamlit + Plotly |

**Supported emotions:** Happy · Sad · Angry · Fearful · Neutral · Calm · Disgust · Surprise

---

## Quick Start

```bash
git clone https://github.com/your-org/ser-platform
cd ser-platform
pip install -r requirements.txt
streamlit run app.py
```

---

## Project Structure

```
ser_platform/
├── app.py                          # Main Streamlit application (10 pages)
├── config.py                       # Global configuration
├── requirements.txt
├── README.md
│
├── datasets/
│   └── dataset_loader.py           # RAVDESS · TESS · EMO-DB loaders
│
├── preprocessing/
│   └── audio_processor.py          # Full preprocessing pipeline
│
├── feature_engineering/
│   ├── feature_extractor.py        # 600+ feature extraction
│   └── visualizations.py           # Plotly charts
│
├── models/
│   └── model_architectures.py      # CNN · LSTM · CNN-LSTM · Transformer
│
├── training/
│   └── trainer.py                  # Training pipeline + cross-validation
│
├── evaluation/
│   └── evaluator.py                # Metrics · confusion matrix · ROC
│
├── nlp/
│   └── speech_nlp.py               # Whisper STT + text emotion
│
├── explainability/
│   └── xai.py                      # SHAP + gradient importance
│
├── reports/
│   └── report_generator.py         # PDF · CSV · Excel export
│
├── utils/
│   ├── helpers.py                   # Utility functions
│   └── ui_components.py             # CSS + HTML components
│
└── saved_models/                    # Auto-saved checkpoints
```

---

## Application Pages

| # | Page | Description |
|---|---|---|
| 1 | **Home** | Platform overview, KPIs, quick-start guide |
| 2 | **Dataset Explorer** | Upload/validate RAVDESS·TESS·EMO-DB, class distribution |
| 3 | **Audio Processing** | Preprocessing pipeline: denoise, VAD, augment |
| 4 | **Feature Extraction** | Waveform, spectrogram, MFCC, mel, feature vector |
| 5 | **Model Training** | Select arch, configure, train, view curves |
| 6 | **Model Evaluation** | Confusion matrix, ROC curves, class-wise F1 |
| 7 | **Real-Time Detection** | Upload or record audio, instant inference + NLP |
| 8 | **Analytics Dashboard** | Unified metrics, radar chart, session timeline |
| 9 | **Report Generation** | Download PDF/CSV/Excel analysis reports |
| 10 | **About Project** | Architecture, tech stack, dataset guide, deployment |

---

## Dataset Setup

### RAVDESS
1. Download from https://zenodo.org/record/1188976
2. Extract — keep the `Actor_XX/` folder structure
3. Upload ZIP in **Dataset Explorer → Upload & Load**

Filename format: `03-01-{emotion}-{intensity}-{statement}-{rep}-{actor}.wav`
Emotion codes: `01`=neutral `02`=calm `03`=happy `04`=sad `05`=angry `06`=fearful `07`=disgust `08`=surprise

### TESS
1. Download from https://tspace.library.utoronto.ca/handle/1807/24487
2. Keep `YAF_*/` and `OAF_*/` folder structure
3. Upload ZIP in **Dataset Explorer**

### EMO-DB
1. Download from http://emodb.bilderbar.info
2. Keep flat `.wav` files
3. Upload ZIP in **Dataset Explorer**

---

## Model Architectures

### 1D CNN
```
Input(604) → Reshape(604,1)
→ Conv1D(64, k=3) → BN → MaxPool → Dropout(0.3)
→ Conv1D(128, k=3) → BN → MaxPool → Dropout(0.3)
→ Conv1D(256, k=3) → BN → MaxPool → Dropout(0.3)
→ GlobalAvgPool → Dense(256) → Dense(128) → Softmax(8)
```

### Bidirectional LSTM + Attention
```
Input(604) → BiLSTM(128) → BiLSTM(64, return_seq=True)
→ Self-Attention → Dense(128) → Softmax(8)
```

### CNN-LSTM Hybrid
```
Input → Conv1D(64) → BN → Conv1D(128) → BN
→ LSTM(128, return_seq=True) → Attention → Dense(128) → Softmax(8)
```

### Transformer (Wav2Vec2)
- Base: `facebook/wav2vec2-base`
- Fine-tuned for 8-class emotion classification
- Processes raw waveforms directly

---

## Feature Extraction (604 dimensions)

| Group | Features | Dimensions |
|---|---|---|
| MFCC | 40 coefficients × {mean, std, min, max} | 160 |
| Delta MFCC | 40 coefficients × 4 stats | 160 |
| Delta² MFCC | 40 coefficients × 4 stats | 160 |
| Spectral | Centroid, BW, Contrast(7), Rolloff × 4 stats | 40 |
| Energy | RMS + ZCR × 4 stats | 8 |
| Chroma | 12 bins × 4 stats | 48 |
| Tonnetz | 6 dimensions × 4 stats | 24 |
| Pitch | F0 × 4 stats | 4 |
| **Total** | | **604** |

---

## Streamlit Cloud Deployment

1. Push to GitHub (public or private repo)
2. Go to [share.streamlit.io](https://share.streamlit.io)
3. **New app** → select repo → `main` → `app.py`
4. Advanced settings → Python **3.11**
5. Click **Deploy** — dependencies install automatically

> **Note:** For Streamlit Cloud's resource limits, TensorFlow CPU is used. The platform degrades gracefully when optional packages (torch, whisper, shap) are unavailable — demo modes activate automatically.

---

## Environment Variables

```bash
LOG_LEVEL=INFO        # DEBUG | INFO | WARNING | ERROR
```

---

## Performance Notes

- **CPU inference:** ~50–200 ms per sample (CNN)
- **GPU inference:** ~5–20 ms per sample
- **Feature extraction:** ~30–80 ms per 3s audio clip
- **Training (RAVDESS full, CNN, 50 epochs):** ~5–15 min CPU, ~1–3 min GPU

---

## License

MIT License — free for academic and commercial use.

---

*Built with ❤️ using Streamlit · TensorFlow · PyTorch · Librosa · HuggingFace*
