# TTS/STT Services - Troubleshooting Guide

> Common issues and solutions for the TTS/STT ML services deployment.

---

## Table of Contents

1. [Connection Issues](#connection-issues)
2. [Deployment Issues](#deployment-issues)
3. [Model Loading Issues](#model-loading-issues)
4. [Audio Format Issues](#audio-format-issues)
5. [Performance Issues](#performance-issues)
6. [ngrok Issues](#ngrok-issues)
7. [Colab-Specific Issues](#colab-specific-issues)
8. [API Response Issues](#api-response-issues)

---

## Connection Issues

### "Connection Refused" Error

**Symptoms:**
```
requests.exceptions.ConnectionError: Connection refused
curl: (7) Failed to connect to xxx.ngrok-free.app port 443
```

**Causes & Solutions:**

1. **Service not running**
   ```bash
   # Check if services are running in Colab
   !ps aux | grep uvicorn
   ```
   Solution: Re-run the deployment cell in Colab.

2. **Wrong URL**
   - Double-check the ngrok URL (they change each deployment)
   - Ensure you're using HTTPS, not HTTP
   - Remove any trailing slashes

3. **ngrok session expired**
   - ngrok free tier sessions expire after ~2 hours
   - Solution: Re-run the ngrok cell to get new URLs

4. **Colab runtime disconnected**
   - Check if Colab shows "Runtime disconnected"
   - Solution: Reconnect and re-run all cells

---

### "SSL Certificate Error"

**Symptoms:**
```
requests.exceptions.SSLError: certificate verify failed
```

**Solutions:**

1. **Disable SSL verification (temporary fix)**
   ```python
   response = requests.post(url, verify=False, ...)
   ```

2. **Update certificates**
   ```bash
   pip install --upgrade certifi
   ```

3. **Use environment variable**
   ```python
   import os
   os.environ['CURL_CA_BUNDLE'] = ''
   ```

---

### "Timeout" Error

**Symptoms:**
```
requests.exceptions.ReadTimeout: Read timed out
```

**Causes & Solutions:**

1. **First request (model warming up)**
   - First request can take 30-60 seconds
   - Increase timeout: `requests.post(url, timeout=120)`

2. **Long audio file**
   - Large files take longer to process
   - Use timeout of 300s for files >1 minute

3. **Network issues**
   - Check your internet connection
   - Try again after a few seconds

---

## Deployment Issues

### "No GPU Available" Warning

**Symptoms:**
```
WARNING: No GPU detected. Running on CPU.
```

**Solution:**
1. Go to Colab menu: **Runtime** > **Change runtime type**
2. Select **T4 GPU** (or any available GPU)
3. Click **Save**
4. Re-run all cells

---

### "CUDA Out of Memory" Error

**Symptoms:**
```
RuntimeError: CUDA out of memory
torch.cuda.OutOfMemoryError
```

**Solutions:**

1. **Restart runtime**
   - Colab menu: **Runtime** > **Restart runtime**
   - Re-run deployment

2. **Clear GPU memory**
   ```python
   import torch
   torch.cuda.empty_cache()
   ```

3. **Process requests sequentially**
   - Don't send parallel requests
   - Wait for each request to complete

4. **Use smaller batch sizes**
   - Process one audio file at a time

---

### "Module Not Found" Error

**Symptoms:**
```
ModuleNotFoundError: No module named 'faster_whisper'
ModuleNotFoundError: No module named 'TTS'
```

**Solution:**
```python
# Re-install dependencies
!pip install faster-whisper TTS torch soundfile librosa
```

---

## Model Loading Issues

### "Model Download Failed"

**Symptoms:**
```
Failed to download model: Connection error
huggingface_hub.utils.HfHubHTTPError
```

**Solutions:**

1. **Check internet connection**
   - Colab should have stable internet

2. **Retry after a few minutes**
   - HuggingFace servers may be temporarily down

3. **Clear cache and retry**
   ```python
   import shutil
   shutil.rmtree('/root/.cache/huggingface', ignore_errors=True)
   ```

---

### "Model Loading Timeout"

**Symptoms:**
- Service hangs during model loading
- No response for >5 minutes

**Solutions:**

1. **Check Colab output**
   - Look for download progress in output cell

2. **Restart and retry**
   - Colab menu: **Runtime** > **Restart runtime**

3. **Check disk space**
   ```python
   !df -h
   ```
   - Models need ~5GB of disk space

---

### "503 Service Unavailable"

**Symptoms:**
```json
{"detail": "STT pipeline not initialized"}
{"detail": "TTS pipeline not initialized"}
```

**Causes:**
- Model is still loading
- Model failed to load

**Solutions:**

1. **Wait for model to load**
   - First startup takes 2-5 minutes

2. **Check service logs**
   ```python
   # In Colab, check output of the cell running the service
   ```

3. **Re-initialize models**
   ```bash
   curl -X POST "$URL/ml/stt/initialize"
   curl -X POST "$URL/ml/tts/initialize"
   ```

---

## Audio Format Issues

### "Audio Preprocessing Failed"

**Symptoms:**
```json
{"detail": "Audio preprocessing failed: ..."}
```

**Causes & Solutions:**

1. **Unsupported format**
   - Supported: WAV, MP3, FLAC, OGG, M4A
   - Convert to WAV: `ffmpeg -i input.xxx -ar 16000 output.wav`

2. **Corrupted file**
   - Try a different audio file
   - Re-record or re-download

3. **Empty file**
   - Check file size: should be >1KB
   - Verify file contains audio data

---

### "Invalid Sample Rate"

**Symptoms:**
```
Warning: Audio sample rate is not 16kHz
```

**Solution:**
- The service auto-resamples, but for best results:
```bash
ffmpeg -i input.wav -ar 16000 -ac 1 output.wav
```

---

### "Audio Too Long"

**Symptoms:**
- Request times out
- CUDA out of memory

**Solutions:**

1. **Split long audio**
   ```python
   from pydub import AudioSegment

   audio = AudioSegment.from_wav("long_audio.wav")
   # Split into 1-minute chunks
   chunk_length = 60 * 1000  # 60 seconds in milliseconds
   chunks = [audio[i:i+chunk_length] for i in range(0, len(audio), chunk_length)]
   ```

2. **Use smaller files**
   - Keep audio under 5 minutes for best results

---

## Performance Issues

### Slow First Request

**This is expected behavior.**

- First request loads models into GPU memory
- Takes 30-60 seconds
- Subsequent requests are faster (2-10 seconds)

**Workaround: Warm up models after deployment**
```python
# Send a dummy request to warm up
requests.post(f"{TTS_URL}/ml/tts/predict",
    json={"text": "warmup", "language": "en"})
```

---

### Slow Transcription

**Causes & Solutions:**

1. **Long audio file**
   - Processing time scales with audio length
   - Expected: ~1 second per 10 seconds of audio

2. **CPU mode**
   - Check if GPU is available
   - GPU is 10-20x faster than CPU

3. **Network latency**
   - ngrok adds some latency (~100-500ms)

---

### Slow TTS Generation

**Causes & Solutions:**

1. **Long text**
   - XTTS v2 is autoregressive (generates word by word)
   - Expected: ~1-2 seconds per 10 words

2. **GPU utilization**
   - Check if another process is using GPU

---

## ngrok Issues

### "ERR_NGROK_108" - Session Limit Reached

**Symptoms:**
```
ERR_NGROK_108: Your account may not run more than 1 tunnel
```

**Solution:**
1. Kill existing ngrok sessions
2. Use different regions for each tunnel:
   ```python
   ngrok.connect(8001, "http", region="us")
   ngrok.connect(8002, "http", region="eu")
   ```

---

### "ERR_NGROK_6022" - Auth Token Invalid

**Symptoms:**
```
ERR_NGROK_6022: Your authtoken is invalid
```

**Solutions:**

1. **Get new auth token**
   - Go to: https://dashboard.ngrok.com/get-started/your-authtoken
   - Copy the new token

2. **Re-authenticate**
   ```python
   ngrok.set_auth_token("your_new_token")
   ```

---

### URLs Keep Changing

**This is expected with ngrok free tier.**

**Solutions:**

1. **Upgrade to ngrok Pro** (paid)
   - Get static URLs

2. **Use environment variables**
   ```python
   import os
   STT_URL = os.getenv("STT_URL", "default_url")
   ```

3. **Update URLs after each deployment**
   - Copy new URLs from Colab output

---

## Colab-Specific Issues

### Runtime Disconnected

**Symptoms:**
- Colab shows "Runtime disconnected"
- All services stop working

**Causes:**
- 90 minutes of inactivity
- Browser tab was closed
- Colab detected no activity

**Solutions:**

1. **Keep-alive script**
   ```python
   from IPython.display import Javascript
   display(Javascript('''
       function ClickConnect() {
           console.log("Keeping alive");
           document.querySelector("colab-connect-button").click();
       }
       setInterval(ClickConnect, 60000);
   '''))
   ```

2. **Keep browser tab active**
   - Don't minimize the tab

3. **Interact periodically**
   - Click in Colab every 30 minutes

---

### "GPU Quota Exceeded"

**Symptoms:**
- No GPU available
- Forced to use CPU

**Solutions:**

1. **Wait 24 hours**
   - Colab free tier has daily GPU quota

2. **Use Colab Pro** (paid)
   - Higher GPU quota

3. **Use different Google account**
   - Each account has separate quota

---

### "Disk Quota Exceeded"

**Symptoms:**
```
OSError: No space left on device
```

**Solutions:**

1. **Clear cache**
   ```python
   !rm -rf /root/.cache/*
   !rm -rf /tmp/*
   ```

2. **Delete old outputs**
   ```python
   !rm -rf /models/tts/*/synthesized/*
   ```

---

## API Response Issues

### "422 Unprocessable Entity"

**Symptoms:**
```json
{"detail": [{"loc": ["body", "text"], "msg": "field required", "type": "value_error.missing"}]}
```

**Causes:**
- Missing required field
- Invalid field type

**Solutions:**

1. **Check request body**
   ```python
   # Correct format
   {
       "text": "Hello",  # Required
       "language": "en"  # Optional
   }
   ```

2. **Check Content-Type header**
   ```python
   headers = {"Content-Type": "application/json"}
   ```

---

### "400 Bad Request"

**Symptoms:**
```json
{"detail": "Uploaded file is empty"}
```

**Solutions:**

1. **Check file exists**
   ```python
   import os
   print(os.path.exists("audio.wav"))
   ```

2. **Check file size**
   ```python
   print(os.path.getsize("audio.wav"))
   ```

3. **Re-upload file**
   - File may be corrupted

---

### Empty or Unexpected Response

**Symptoms:**
- Empty response body
- ngrok HTML page instead of JSON

**Solutions:**

1. **Add ngrok header**
   ```python
   headers = {"ngrok-skip-browser-warning": "true"}
   ```

2. **Check URL**
   - Should end with `/ml/stt/transcribe` not just the base URL

---

## Quick Diagnostic Commands

### Check Service Status

```python
import requests

def check_services(stt_url, tts_url):
    """Quick health check for both services."""
    headers = {"ngrok-skip-browser-warning": "true"}

    try:
        stt = requests.get(f"{stt_url}/ml/stt/health", headers=headers, timeout=10)
        print(f"STT: {stt.status_code} - {stt.json().get('status', 'unknown')}")
    except Exception as e:
        print(f"STT: Error - {e}")

    try:
        tts = requests.get(f"{tts_url}/ml/tts/health", headers=headers, timeout=10)
        print(f"TTS: {tts.status_code} - {tts.json().get('status', 'unknown')}")
    except Exception as e:
        print(f"TTS: Error - {e}")

check_services("https://xxx.ngrok-free.app", "https://yyy.ngrok-free.app")
```

### Check GPU Memory

```python
# Run in Colab
!nvidia-smi
```

### Check Disk Space

```python
# Run in Colab
!df -h
```

### Check Running Processes

```python
# Run in Colab
!ps aux | grep -E "uvicorn|python|ngrok"
```

---

## Still Having Issues?

1. **Check Colab output logs** for error messages
2. **Restart runtime** and re-deploy
3. **Try a different browser** (Chrome recommended)
4. **Check ngrok dashboard** at https://dashboard.ngrok.com/
5. **File an issue** at https://github.com/shaivya1404/tts-stt/issues

---

**Happy Debugging!**
