# TTS/STT ML Services - Google Colab Deployment Guide

> **One-Click Deployment** of Text-to-Speech and Speech-to-Text services on Google Colab with free GPU acceleration.

---

## What This Does

This deployment script sets up two production-ready ML services on Google Colab:

| Service | Model | Port | Capabilities |
|---------|-------|------|--------------|
| **STT** (Speech-to-Text) | Faster-Whisper large-v3 | 8002 | Transcribes audio to text in 99+ languages |
| **TTS** (Text-to-Speech) | XTTS v2 (Coqui TTS) | 8001 | Synthesizes natural speech in 17+ languages |

Both services are exposed via **ngrok tunnels**, giving you public HTTPS URLs accessible from anywhere.

---

## Prerequisites

Before you begin, you'll need:

### 1. Google Account
- Free Google account for Colab access
- No credit card required

### 2. ngrok Account (Free)
- Sign up at [ngrok.com](https://ngrok.com)
- Get your **auth token** (required for public URLs)

### 3. Optional: Sample Audio Files
- For testing STT, have a `.wav` or `.mp3` file ready
- Recommended: 16kHz sample rate, mono channel

---

## How to Get Your ngrok Auth Token

1. Go to [dashboard.ngrok.com/signup](https://dashboard.ngrok.com/signup)
2. Create a free account (or sign in with Google/GitHub)
3. Navigate to **Your Authtoken** in the left sidebar
4. Click **Copy** to copy your token
5. Save it somewhere safe - you'll need it during deployment

Your token looks like: `2abc123XYZ_abcdefghijklmnop`

---

## Step-by-Step Deployment

### Step 1: Open the Deployment Notebook

Click the button below to open the deployment notebook in Google Colab:

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/shaivya1404/tts-stt/blob/main/colab_deployment.ipynb)

Or manually:
1. Go to [colab.research.google.com](https://colab.research.google.com)
2. Click **File** > **Open Notebook**
3. Select **GitHub** tab
4. Enter: `https://github.com/shaivya1404/tts-stt`
5. Open `colab_deployment.ipynb`

### Step 2: Enable GPU Runtime

**This is critical for performance!**

1. Click **Runtime** in the menu bar
2. Click **Change runtime type**
3. Under **Hardware accelerator**, select **T4 GPU** (or any available GPU)
4. Click **Save**

### Step 3: Run the Deployment Cell

1. Click the **Play button** (or press `Shift+Enter`) on the deployment cell
2. When prompted, enter your **ngrok auth token**
3. Wait 3-5 minutes for models to download and services to start

### Step 4: Get Your Service URLs

After successful deployment, you'll see output like:

```
============================================================
DEPLOYMENT SUCCESSFUL!
============================================================

STT Service (Speech-to-Text):
  URL: https://abc123.ngrok-free.app
  Health: https://abc123.ngrok-free.app/ml/stt/health

TTS Service (Text-to-Speech):
  URL: https://xyz789.ngrok-free.app
  Health: https://xyz789.ngrok-free.app/ml/tts/health

============================================================
```

**Save these URLs!** They're your API endpoints.

---

## Service Endpoints

### STT Service (Speech-to-Text)

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/ml/stt/health` | GET | Health check |
| `/ml/stt/models` | GET | List loaded models |
| `/ml/stt/transcribe` | POST | Transcribe audio file |

### TTS Service (Text-to-Speech)

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/ml/tts/health` | GET | Health check |
| `/ml/tts/models` | GET | List loaded models |
| `/ml/tts/predict` | POST | Synthesize speech |

---

## Quick Test Examples

### Test STT (Speech-to-Text)

```bash
# Check health
curl https://YOUR_STT_URL.ngrok-free.app/ml/stt/health

# Transcribe audio
curl -X POST https://YOUR_STT_URL.ngrok-free.app/ml/stt/transcribe \
  -F "file=@your_audio.wav" \
  -F "language_hint=en"
```

### Test TTS (Text-to-Speech)

```bash
# Check health
curl https://YOUR_TTS_URL.ngrok-free.app/ml/tts/health

# Generate speech
curl -X POST https://YOUR_TTS_URL.ngrok-free.app/ml/tts/predict \
  -H "Content-Type: application/json" \
  -d '{"text": "Hello, world!", "language": "en"}' \
  --output speech.wav
```

---

## Supported Languages

### STT (Faster-Whisper) - 99+ Languages
Most common: English, Hindi, Tamil, Telugu, Bengali, Marathi, Gujarati, Kannada, Malayalam, Punjabi, Spanish, French, German, Chinese, Japanese, Korean, Arabic, Russian

### TTS (XTTS v2) - 17 Languages
English, Hindi, Spanish, French, German, Italian, Portuguese, Polish, Turkish, Russian, Dutch, Czech, Arabic, Chinese, Japanese, Korean, Hungarian

---

## Response Formats

### STT Response
```json
{
  "text": "Hello, how are you today?",
  "language": "en",
  "confidence": 0.95,
  "timestamps": [
    {"start": 0.0, "end": 0.5, "word": "Hello"},
    {"start": 0.5, "end": 0.8, "word": "how"},
    {"start": 0.8, "end": 1.0, "word": "are"},
    {"start": 1.0, "end": 1.2, "word": "you"},
    {"start": 1.2, "end": 1.6, "word": "today"}
  ],
  "meta": {
    "duration_seconds": 1.6,
    "model_name": "whisper_large-v3"
  },
  "modelUsed": "whisper_large-v3:faster-whisper"
}
```

### TTS Response
```json
{
  "audio_path": "/models/tts/xtts_v2/v1/synthesized/default_abc123.wav",
  "duration": 2.5,
  "status": "success",
  "meta": {
    "language": "en",
    "speed": 1.0,
    "mos_score": 4.0,
    "model": "xtts_v2"
  }
}
```

---

## Important Limitations

### Colab Session Limits
- **Free tier**: Sessions timeout after ~90 minutes of inactivity
- **GPU quota**: ~4 hours/day of GPU time on free tier
- **Session disconnect**: You'll need to redeploy after disconnect

### ngrok Free Tier Limits
- **1 tunnel** at a time per region (we use 2 regions)
- **Sessions expire** after 2 hours (need to restart)
- **Random URLs** - URLs change each deployment

### Performance Notes
- **First request**: May take 30-60 seconds (model warming up)
- **Subsequent requests**: 2-10 seconds depending on input length
- **GPU memory**: ~14GB used (10GB Whisper + 4GB XTTS)

---

## Keeping Services Alive

To prevent Colab from disconnecting:

1. **Keep the tab active** - Don't minimize or switch tabs for long
2. **Use the keep-alive script** in the notebook
3. **Check every 30 minutes** - Colab may ask "Are you still there?"

### Auto Keep-Alive Code (add to notebook)
```python
import time
from IPython.display import display, Javascript

def keep_alive():
    display(Javascript('function ClickConnect(){console.log("Keeping alive"); document.querySelector("colab-connect-button").click()}setInterval(ClickConnect, 60000)'))

keep_alive()
```

---

## FAQ

### Q: Why do URLs change every deployment?
**A:** ngrok free tier generates random URLs. Upgrade to ngrok paid plan for custom domains.

### Q: Can I use this in production?
**A:** This is for development/testing. For production, deploy on dedicated infrastructure (AWS, GCP, Azure).

### Q: How long can I use it?
**A:** Colab free tier: ~4 hours GPU/day. ngrok: Sessions expire after 2 hours.

### Q: Why is the first request slow?
**A:** Models are lazily loaded on first request. Subsequent requests are faster.

### Q: Can I process batch requests?
**A:** Yes, but process sequentially. Parallel requests may cause OOM errors.

### Q: What audio formats are supported?
**A:** STT supports: WAV, MP3, FLAC, OGG, M4A. TTS outputs: WAV.

### Q: Can I use my own voice for TTS?
**A:** Yes! Provide a `speaker_wav` reference audio file for voice cloning.

### Q: Why did my session disconnect?
**A:** Colab disconnects after ~90 minutes of inactivity. Redeploy when needed.

---

## Next Steps

1. **Read the Testing Guide**: `TESTING_GUIDE.md` for comprehensive testing examples
2. **Run Test Scripts**: Use `test_services.py` for automated testing
3. **Import Postman Collection**: `postman_collection.json` for API testing
4. **Check Troubleshooting**: `TROUBLESHOOTING.md` if you encounter issues

---

## Support

- **GitHub Issues**: [github.com/shaivya1404/tts-stt/issues](https://github.com/shaivya1404/tts-stt/issues)
- **Documentation**: See `/docs` folder for detailed guides

---

## License

This project is for educational and development purposes. See LICENSE file for details.

---

**Happy Testing!**
