# TTS & STT Project - Master Context File

> **Purpose**: This file captures all project context so you can continue anytime.
> **Last Updated**: 2026-02-02

---

## Project Goal

Build production-quality **Text-to-Speech (TTS)** and **Speech-to-Text (STT)** for:
- **Languages**: English + Hindi
- **Voice Quality**: Human-like (NOT robotic)
- **Differentiator**: Custom dataset (not public data like college projects)

---

## Current Status

| Component | Status | Notes |
|-----------|--------|-------|
| **STT Dataset** | ✅ Done | 4+ hours collected, 1334 segments |
| **STT Model** | ✅ **TRAINED** | Whisper Small + LoRA, Loss: 1.59→0.97 |
| **TTS Dataset** | ⏳ Pending | Need 30-sec reference audio |
| **TTS Model** | ⏳ Ready to start | XTTS v2 voice cloning |
| Deployment | Ready | Docker configs exist |

### STT Training Completed (February 2026)
- **Platform**: Kaggle (Free GPU - P100)
- **Time**: 24 minutes
- **Loss**: 1.594 → 0.971 (39% improvement)
- **Model**: whisper-stt-finetuned (LoRA weights)
- **Report**: See `TRAINING_REPORT.md` for full details

---

## TTS - Next Phase

### Approach: XTTS v2 Voice Cloning
- **No training required** - Just need 30-second reference audio
- XTTS v2 clones voice instantly
- Supports Hindi + English

### What You Need
| Item | Requirement |
|------|-------------|
| Reference Audio | 30 seconds of YOUR voice |
| Quality | Clear, no background noise |
| Format | WAV (preferred) or MP3 |
| Content | Natural speech (read a paragraph) |

### Sample Text to Record
> "Hello, my name is [your name]. I am recording this audio to create a text to speech model. This sample will be used as a reference for voice cloning. The quality of the generated speech depends on this recording, so I am speaking clearly and naturally. Thank you for listening to this demonstration."

### TTS Steps
1. **Record 30 seconds** of your voice reading the sample text
2. **Save as** `data/tts/reference/reference_voice.wav`
3. **Upload to Kaggle** as dataset `tts-reference-audio`
4. **Run notebook** `notebooks/kaggle_tts_xtts.ipynb`
5. **Generate speech** in your cloned voice!

### TTS Notebook Location
```
notebooks/kaggle_tts_xtts.ipynb
```

### TTS Code (Quick Test)
```python
from TTS.api import TTS
import torch

# Load model
device = "cuda" if torch.cuda.is_available() else "cpu"
tts = TTS("tts_models/multilingual/multi-dataset/xtts_v2").to(device)

# Generate speech
tts.tts_to_file(
    text="Hello, this is my cloned voice!",
    file_path="output.wav",
    speaker_wav="reference_voice.wav",
    language="en"
)

# Hindi
tts.tts_to_file(
    text="नमस्ते, यह मेरी क्लोन आवाज़ है!",
    file_path="output_hindi.wav",
    speaker_wav="reference_voice.wav",
    language="hi"
)
```

---

## STT - Completed

### Training Results
| Metric | Value |
|--------|-------|
| Base Model | Whisper Small |
| Method | LoRA Fine-tuning |
| Training Loss Start | 1.594 |
| Training Loss End | 0.971 |
| Improvement | 39% |
| Training Time | 24 minutes |
| Platform | Kaggle P100 GPU |
| Cost | $0 (Free) |

### STT Model Usage
```python
from peft import PeftModel
from transformers import WhisperForConditionalGeneration, WhisperProcessor
import librosa, torch

# Load model
base = WhisperForConditionalGeneration.from_pretrained("openai/whisper-small")
model = PeftModel.from_pretrained(base, "./whisper-stt-finetuned")
processor = WhisperProcessor.from_pretrained("./whisper-stt-finetuned")

# Transcribe
audio, sr = librosa.load("audio.wav", sr=16000)
inputs = processor(audio, sampling_rate=16000, return_tensors="pt")
with torch.no_grad():
    ids = model.generate(inputs.input_features, max_length=225)
text = processor.batch_decode(ids, skip_special_tokens=True)[0]
print(text)
```

### STT Data Summary
| Language | Segments | Duration |
|----------|----------|----------|
| Hindi | 1,151 | ~3.5 hours |
| English | 300 | ~45 min |
| **Total** | **1,451** | **~4.3 hours** |

---

## All Notebooks

| Notebook | Purpose | Status |
|----------|---------|--------|
| `kaggle_stt_training_v2.ipynb` | STT training (fixed) | ✅ Tested |
| `kaggle_tts_xtts.ipynb` | TTS voice cloning | ✅ Ready |

---

## All Documentation

| File | Purpose |
|------|---------|
| `PROJECT_CONTEXT.md` | Master context (this file) |
| `TRAINING_REPORT.md` | Detailed STT training report |
| `GCP_GPU_GUIDE.md` | GCP setup guide (if needed) |

---

## File Structure

```
tts-stt/
├── PROJECT_CONTEXT.md          <- This file (READ FIRST)
├── TRAINING_REPORT.md          <- STT training details
├── GCP_GPU_GUIDE.md            <- Cloud GPU setup
├── data/
│   ├── stt/
│   │   ├── extracted/          <- Raw audio segments
│   │   └── combined/           <- Training manifests
│   └── tts/
│       └── reference/          <- Put reference_voice.wav here
├── ml-service/
│   ├── training/
│   │   ├── stt/                <- STT training scripts
│   │   └── tts/                <- TTS training scripts
│   ├── stt-service/            <- STT inference service
│   └── tts-service/            <- TTS inference service
└── notebooks/
    ├── kaggle_stt_training_v2.ipynb  <- STT training
    └── kaggle_tts_xtts.ipynb         <- TTS voice cloning
```

---

## Next Steps (In Order)

### ✅ COMPLETED
- [x] Collect STT data (4+ hours)
- [x] Train STT model on Kaggle
- [x] Create TTS notebook

### 🔄 IN PROGRESS
- [ ] **Record 30-second reference audio**
- [ ] Upload to Kaggle
- [ ] Run TTS notebook
- [ ] Test voice cloning

### 📋 TODO
- [ ] Evaluate STT model (WER/CER)
- [ ] Collect more STT data (20+ hours)
- [ ] Build API endpoints
- [ ] Docker deployment
- [ ] Production setup

---

## Resume Instructions

### To Continue TTS Work:
> "Continue TTS-STT project. Read PROJECT_CONTEXT.md. I want to work on TTS."

### To Continue STT Work:
> "Continue TTS-STT project. Read PROJECT_CONTEXT.md. I want to improve STT."

### To Deploy:
> "Continue TTS-STT project. Read PROJECT_CONTEXT.md. I want to deploy."

---

## Session Log

### 2026-02-02
- ✅ Completed STT model training on Kaggle
- ✅ Training loss: 1.59 → 0.97 (39% improvement)
- ✅ Created TRAINING_REPORT.md with full documentation
- ✅ Created TTS notebook (kaggle_tts_xtts.ipynb)
- ✅ Updated all context files
- **Next**: Record reference audio for TTS

### 2026-01-27
- Decided: STT first, TTS later
- Created extraction pipeline
- Extracted 1,334 audio segments (~4 hours)
- Combined manifests ready for training

---

## Technical Notes

### STT
- **Model**: Whisper Small + LoRA adapters
- **Sample Rate**: 16kHz
- **Max Label Length**: 440 tokens
- **Training**: 3 epochs, batch size 8

### TTS
- **Model**: XTTS v2 (Coqui TTS)
- **Sample Rate**: 22.05kHz
- **Reference Audio**: 10-30 seconds recommended
- **Languages**: en, hi (+ 15 more supported)

### Kaggle Tips
- Always enable GPU before running
- Use `processing_class` instead of `tokenizer` (new API)
- Filter labels > 440 tokens to avoid errors
- Use direct librosa loading instead of datasets.Audio

---

## Quick Commands

### Record Reference Audio (Windows)
```
1. Open Voice Recorder app
2. Record 30 seconds reading sample text
3. Save to: data/tts/reference/reference_voice.wav
```

### Test STT Model
```python
# Quick test
from peft import PeftModel
from transformers import WhisperForConditionalGeneration, WhisperProcessor

base = WhisperForConditionalGeneration.from_pretrained("openai/whisper-small")
model = PeftModel.from_pretrained(base, "./whisper-stt-finetuned")
processor = WhisperProcessor.from_pretrained("./whisper-stt-finetuned")
```

### Test TTS Model
```python
from TTS.api import TTS
tts = TTS("tts_models/multilingual/multi-dataset/xtts_v2")
tts.tts_to_file(text="Hello!", file_path="out.wav", speaker_wav="ref.wav", language="en")
```
