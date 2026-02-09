# TTS/STT Services - Complete Testing Guide

> Comprehensive testing documentation with examples for Python, cURL, and Postman.

---

## Table of Contents

1. [Quick Start](#quick-start)
2. [STT Testing (Speech-to-Text)](#stt-testing-speech-to-text)
3. [TTS Testing (Text-to-Speech)](#tts-testing-text-to-speech)
4. [Integration Testing](#integration-testing)
5. [Performance Testing](#performance-testing)
6. [Error Handling](#error-handling)
7. [Multi-Language Testing](#multi-language-testing)

---

## Quick Start

### Set Your Service URLs

Before testing, set your ngrok URLs as environment variables:

**Windows (PowerShell):**
```powershell
$env:STT_URL = "https://your-stt-url.ngrok-free.app"
$env:TTS_URL = "https://your-tts-url.ngrok-free.app"
```

**Windows (CMD):**
```cmd
set STT_URL=https://your-stt-url.ngrok-free.app
set TTS_URL=https://your-tts-url.ngrok-free.app
```

**Linux/Mac:**
```bash
export STT_URL="https://your-stt-url.ngrok-free.app"
export TTS_URL="https://your-tts-url.ngrok-free.app"
```

### Quick Health Check

```bash
# Check STT health
curl $STT_URL/ml/stt/health

# Check TTS health
curl $TTS_URL/ml/tts/health
```

**Expected Response:**
```json
{
  "status": "ok",
  "detail": "stt-service healthy",
  "models": [{"name": "whisper_large-v3", "status": "ready"}]
}
```

---

## STT Testing (Speech-to-Text)

### 1. Basic English Transcription

#### Using cURL
```bash
curl -X POST "$STT_URL/ml/stt/transcribe" \
  -H "ngrok-skip-browser-warning: true" \
  -F "file=@sample_english.wav" \
  -F "language_hint=en"
```

#### Using Python
```python
import requests

STT_URL = "https://your-stt-url.ngrok-free.app"

def transcribe_audio(file_path, language_hint="en"):
    """Transcribe an audio file to text."""
    url = f"{STT_URL}/ml/stt/transcribe"

    headers = {"ngrok-skip-browser-warning": "true"}

    with open(file_path, "rb") as f:
        files = {"file": (file_path, f, "audio/wav")}
        data = {"language_hint": language_hint}

        response = requests.post(url, files=files, data=data, headers=headers)

    if response.status_code == 200:
        result = response.json()
        print(f"Text: {result['text']}")
        print(f"Language: {result['language']}")
        print(f"Confidence: {result['confidence']}")
        return result
    else:
        print(f"Error: {response.status_code} - {response.text}")
        return None

# Usage
result = transcribe_audio("sample_english.wav", "en")
```

**Expected Response:**
```json
{
  "text": "Hello, this is a test of the speech to text system.",
  "language": "en",
  "confidence": 0.94,
  "timestamps": [
    {"start": 0.0, "end": 0.48, "word": "Hello"},
    {"start": 0.48, "end": 0.72, "word": "this"},
    {"start": 0.72, "end": 0.88, "word": "is"},
    {"start": 0.88, "end": 0.96, "word": "a"},
    {"start": 0.96, "end": 1.28, "word": "test"}
  ],
  "meta": {
    "duration_seconds": 3.5,
    "quality_score": 0.92,
    "model_name": "whisper_large-v3"
  },
  "modelUsed": "whisper_large-v3:primary",
  "status": "success"
}
```

---

### 2. Hindi Audio Transcription

#### Using cURL
```bash
curl -X POST "$STT_URL/ml/stt/transcribe" \
  -H "ngrok-skip-browser-warning: true" \
  -F "file=@hindi_sample.wav" \
  -F "language_hint=hi"
```

#### Using Python
```python
import requests

def transcribe_hindi(file_path):
    """Transcribe Hindi audio."""
    STT_URL = "https://your-stt-url.ngrok-free.app"

    with open(file_path, "rb") as f:
        response = requests.post(
            f"{STT_URL}/ml/stt/transcribe",
            files={"file": f},
            data={"language_hint": "hi"},
            headers={"ngrok-skip-browser-warning": "true"}
        )

    result = response.json()
    print(f"Hindi Text: {result['text']}")
    # Output: "नमस्ते, यह एक परीक्षण है।"
    return result

transcribe_hindi("hindi_sample.wav")
```

---

### 3. Auto Language Detection

Let Whisper auto-detect the language:

```python
import requests

def transcribe_auto_detect(file_path):
    """Transcribe with automatic language detection."""
    STT_URL = "https://your-stt-url.ngrok-free.app"

    with open(file_path, "rb") as f:
        response = requests.post(
            f"{STT_URL}/ml/stt/transcribe",
            files={"file": f},
            # No language_hint - auto-detect
            headers={"ngrok-skip-browser-warning": "true"}
        )

    result = response.json()
    print(f"Detected Language: {result['language']}")
    print(f"Text: {result['text']}")
    return result

transcribe_auto_detect("unknown_language.wav")
```

---

### 4. Different Audio Formats

```python
import requests

STT_URL = "https://your-stt-url.ngrok-free.app"

def transcribe_any_format(file_path):
    """Transcribe audio in any supported format."""

    # Determine content type based on extension
    extension = file_path.split(".")[-1].lower()
    content_types = {
        "wav": "audio/wav",
        "mp3": "audio/mpeg",
        "flac": "audio/flac",
        "ogg": "audio/ogg",
        "m4a": "audio/mp4",
        "webm": "audio/webm"
    }

    content_type = content_types.get(extension, "audio/wav")

    with open(file_path, "rb") as f:
        files = {"file": (file_path, f, content_type)}
        response = requests.post(
            f"{STT_URL}/ml/stt/transcribe",
            files=files,
            headers={"ngrok-skip-browser-warning": "true"}
        )

    return response.json()

# Test different formats
for audio_file in ["test.wav", "test.mp3", "test.flac", "test.m4a"]:
    try:
        result = transcribe_any_format(audio_file)
        print(f"{audio_file}: {result['text'][:50]}...")
    except FileNotFoundError:
        print(f"{audio_file}: File not found")
```

---

### 5. Long Audio File (>1 minute)

```python
import requests
import time

def transcribe_long_audio(file_path, timeout=300):
    """Transcribe long audio files with extended timeout."""
    STT_URL = "https://your-stt-url.ngrok-free.app"

    print(f"Transcribing {file_path}...")
    start_time = time.time()

    with open(file_path, "rb") as f:
        response = requests.post(
            f"{STT_URL}/ml/stt/transcribe",
            files={"file": f},
            data={"language_hint": "en"},
            headers={"ngrok-skip-browser-warning": "true"},
            timeout=timeout  # 5 minutes timeout for long files
        )

    elapsed = time.time() - start_time
    result = response.json()

    print(f"Transcription completed in {elapsed:.2f} seconds")
    print(f"Audio duration: {result['meta']['duration_seconds']}s")
    print(f"Text length: {len(result['text'])} characters")
    print(f"Word count: {len(result['timestamps'])} words")

    return result

result = transcribe_long_audio("long_audio_5min.wav")
print(result['text'])
```

---

### 6. Batch Transcription

```python
import requests
import os
from concurrent.futures import ThreadPoolExecutor, as_completed

STT_URL = "https://your-stt-url.ngrok-free.app"

def transcribe_single(file_path):
    """Transcribe a single file."""
    with open(file_path, "rb") as f:
        response = requests.post(
            f"{STT_URL}/ml/stt/transcribe",
            files={"file": f},
            headers={"ngrok-skip-browser-warning": "true"},
            timeout=120
        )
    return file_path, response.json()

def batch_transcribe(audio_files, max_workers=2):
    """Transcribe multiple files.

    Note: Use max_workers=2 to avoid GPU memory issues.
    """
    results = {}

    # Sequential processing recommended for Colab
    for file_path in audio_files:
        try:
            _, result = transcribe_single(file_path)
            results[file_path] = result
            print(f"✓ {file_path}: {result['text'][:50]}...")
        except Exception as e:
            results[file_path] = {"error": str(e)}
            print(f"✗ {file_path}: {e}")

    return results

# Usage
audio_files = ["audio1.wav", "audio2.wav", "audio3.wav"]
results = batch_transcribe(audio_files)

# Save results
import json
with open("transcription_results.json", "w", encoding="utf-8") as f:
    json.dump(results, f, indent=2, ensure_ascii=False)
```

---

## TTS Testing (Text-to-Speech)

### 1. Simple English Text

#### Using cURL
```bash
curl -X POST "$TTS_URL/ml/tts/predict" \
  -H "Content-Type: application/json" \
  -H "ngrok-skip-browser-warning: true" \
  -d '{
    "text": "Hello, welcome to our text to speech service.",
    "language": "en",
    "speed": 1.0
  }'
```

**Note:** The response contains `audio_path` on the server. To get the actual audio, you need to download it or use a modified endpoint.

#### Using Python (with audio download)
```python
import requests
import json

TTS_URL = "https://your-tts-url.ngrok-free.app"

def synthesize_speech(text, language="en", speed=1.0, output_file="output.wav"):
    """Generate speech from text and save to file."""

    payload = {
        "text": text,
        "language": language,
        "speed": speed
    }

    headers = {
        "Content-Type": "application/json",
        "ngrok-skip-browser-warning": "true"
    }

    response = requests.post(
        f"{TTS_URL}/ml/tts/predict",
        json=payload,
        headers=headers
    )

    if response.status_code == 200:
        result = response.json()
        print(f"Status: {result['status']}")
        print(f"Duration: {result['duration']} seconds")
        print(f"Audio path (server): {result['audio_path']}")
        return result
    else:
        print(f"Error: {response.status_code} - {response.text}")
        return None

# Usage
result = synthesize_speech(
    text="Hello, this is a test of the text to speech system.",
    language="en",
    speed=1.0
)
```

**Expected Response:**
```json
{
  "audio_path": "/models/tts/xtts_v2/v1/synthesized/default_abc123def.wav",
  "duration": 3.2,
  "status": "success",
  "meta": {
    "language": "en",
    "speed": 1.0,
    "mos_score": 4.0,
    "model": "xtts_v2",
    "char_count": 52
  }
}
```

---

### 2. Hindi Text Synthesis

```python
import requests

def synthesize_hindi(text):
    """Generate Hindi speech."""
    TTS_URL = "https://your-tts-url.ngrok-free.app"

    payload = {
        "text": text,
        "language": "hi",
        "speed": 1.0
    }

    response = requests.post(
        f"{TTS_URL}/ml/tts/predict",
        json=payload,
        headers={
            "Content-Type": "application/json",
            "ngrok-skip-browser-warning": "true"
        }
    )

    result = response.json()
    print(f"Hindi TTS Result: {result}")
    return result

# Usage
synthesize_hindi("नमस्ते, आप कैसे हैं? यह एक परीक्षण है।")
```

---

### 3. Multi-Language Examples

```python
import requests

TTS_URL = "https://your-tts-url.ngrok-free.app"

# Test texts in different languages
test_texts = {
    "en": "Hello, how are you today?",
    "hi": "नमस्ते, आप कैसे हैं?",
    "es": "Hola, cómo estás hoy?",
    "fr": "Bonjour, comment allez-vous?",
    "de": "Hallo, wie geht es Ihnen?",
    "ta": "வணக்கம், நீங்கள் எப்படி இருக்கிறீர்கள்?",
    "te": "హలో, మీరు ఎలా ఉన్నారు?",
    "ja": "こんにちは、お元気ですか？",
    "zh-cn": "你好，你今天怎么样？",
    "ar": "مرحبا، كيف حالك اليوم؟"
}

def test_all_languages():
    """Test TTS in multiple languages."""
    results = {}

    for lang, text in test_texts.items():
        try:
            response = requests.post(
                f"{TTS_URL}/ml/tts/predict",
                json={"text": text, "language": lang},
                headers={
                    "Content-Type": "application/json",
                    "ngrok-skip-browser-warning": "true"
                },
                timeout=60
            )

            if response.status_code == 200:
                result = response.json()
                results[lang] = {
                    "status": "success",
                    "duration": result.get("duration"),
                    "text_length": len(text)
                }
                print(f"✓ {lang}: {text[:30]}... ({result.get('duration')}s)")
            else:
                results[lang] = {"status": "error", "code": response.status_code}
                print(f"✗ {lang}: Error {response.status_code}")

        except Exception as e:
            results[lang] = {"status": "error", "message": str(e)}
            print(f"✗ {lang}: {e}")

    return results

results = test_all_languages()
```

---

### 4. Speed Control Test

```python
import requests

TTS_URL = "https://your-tts-url.ngrok-free.app"

def test_speed_variations():
    """Test different speech speeds."""
    text = "Testing speech synthesis at different speeds."
    speeds = [0.5, 0.75, 1.0, 1.25, 1.5, 2.0]

    for speed in speeds:
        response = requests.post(
            f"{TTS_URL}/ml/tts/predict",
            json={"text": text, "language": "en", "speed": speed},
            headers={
                "Content-Type": "application/json",
                "ngrok-skip-browser-warning": "true"
            }
        )

        result = response.json()
        print(f"Speed {speed}x: Duration = {result.get('duration')}s")

test_speed_variations()
```

**Expected Output:**
```
Speed 0.5x: Duration = 6.4s
Speed 0.75x: Duration = 4.3s
Speed 1.0x: Duration = 3.2s
Speed 1.25x: Duration = 2.6s
Speed 1.5x: Duration = 2.1s
Speed 2.0x: Duration = 1.6s
```

---

### 5. Long Paragraph Synthesis

```python
import requests

def synthesize_long_text(text, language="en"):
    """Synthesize long text (may take longer)."""
    TTS_URL = "https://your-tts-url.ngrok-free.app"

    print(f"Synthesizing {len(text)} characters...")

    response = requests.post(
        f"{TTS_URL}/ml/tts/predict",
        json={"text": text, "language": language},
        headers={
            "Content-Type": "application/json",
            "ngrok-skip-browser-warning": "true"
        },
        timeout=180  # 3 minutes for long texts
    )

    result = response.json()
    print(f"Generated {result.get('duration')}s of audio")
    return result

# Long paragraph example
long_text = """
Artificial intelligence has transformed the way we interact with technology.
From voice assistants to autonomous vehicles, AI is becoming an integral part
of our daily lives. Natural language processing, a subset of AI, enables
machines to understand and respond to human language in meaningful ways.
Text-to-speech technology is a prime example of this, converting written
text into natural-sounding speech that can be used for accessibility,
education, and entertainment purposes.
"""

result = synthesize_long_text(long_text.strip())
```

---

### 6. Batch TTS Generation

```python
import requests
import json

TTS_URL = "https://your-tts-url.ngrok-free.app"

def batch_synthesize(texts, language="en"):
    """Generate speech for multiple texts."""
    results = []

    for i, text in enumerate(texts):
        print(f"Processing {i+1}/{len(texts)}: {text[:30]}...")

        response = requests.post(
            f"{TTS_URL}/ml/tts/predict",
            json={"text": text, "language": language},
            headers={
                "Content-Type": "application/json",
                "ngrok-skip-browser-warning": "true"
            }
        )

        result = response.json()
        results.append({
            "text": text,
            "duration": result.get("duration"),
            "status": result.get("status"),
            "audio_path": result.get("audio_path")
        })

    return results

# Example: Generate multiple greetings
greetings = [
    "Good morning! Welcome to our service.",
    "Thank you for calling. How may I help you?",
    "Your order has been confirmed.",
    "Please hold while we connect you.",
    "Goodbye and have a great day!"
]

results = batch_synthesize(greetings)

# Save results
with open("tts_batch_results.json", "w") as f:
    json.dump(results, f, indent=2)
```

---

## Integration Testing

### 1. STT → TTS Round Trip

Test the complete pipeline: Audio → Text → Audio

```python
import requests
import time

STT_URL = "https://your-stt-url.ngrok-free.app"
TTS_URL = "https://your-tts-url.ngrok-free.app"

def round_trip_test(audio_file, language="en"):
    """Complete round-trip: Audio → Text → Audio."""

    print("=" * 50)
    print("ROUND TRIP TEST")
    print("=" * 50)

    # Step 1: Transcribe audio to text
    print("\n[1] Transcribing audio to text...")
    start = time.time()

    with open(audio_file, "rb") as f:
        stt_response = requests.post(
            f"{STT_URL}/ml/stt/transcribe",
            files={"file": f},
            data={"language_hint": language},
            headers={"ngrok-skip-browser-warning": "true"}
        )

    stt_time = time.time() - start
    stt_result = stt_response.json()

    print(f"   Transcribed text: {stt_result['text']}")
    print(f"   Language: {stt_result['language']}")
    print(f"   Confidence: {stt_result['confidence']}")
    print(f"   Time: {stt_time:.2f}s")

    # Step 2: Synthesize text to audio
    print("\n[2] Synthesizing text to audio...")
    start = time.time()

    tts_response = requests.post(
        f"{TTS_URL}/ml/tts/predict",
        json={
            "text": stt_result['text'],
            "language": stt_result['language']
        },
        headers={
            "Content-Type": "application/json",
            "ngrok-skip-browser-warning": "true"
        }
    )

    tts_time = time.time() - start
    tts_result = tts_response.json()

    print(f"   Audio duration: {tts_result['duration']}s")
    print(f"   Status: {tts_result['status']}")
    print(f"   Time: {tts_time:.2f}s")

    # Summary
    print("\n" + "=" * 50)
    print("SUMMARY")
    print("=" * 50)
    print(f"Total time: {stt_time + tts_time:.2f}s")
    print(f"STT time: {stt_time:.2f}s")
    print(f"TTS time: {tts_time:.2f}s")

    return {
        "stt_result": stt_result,
        "tts_result": tts_result,
        "total_time": stt_time + tts_time
    }

# Run the test
result = round_trip_test("sample_audio.wav", "en")
```

---

### 2. Multi-Language Pipeline

```python
import requests

STT_URL = "https://your-stt-url.ngrok-free.app"
TTS_URL = "https://your-tts-url.ngrok-free.app"

def translate_speech(audio_file, source_lang, target_lang, target_text):
    """
    Simulate speech translation:
    1. Transcribe source audio
    2. (Translation would happen here - not implemented)
    3. Synthesize in target language
    """

    # Step 1: Transcribe
    with open(audio_file, "rb") as f:
        stt_response = requests.post(
            f"{STT_URL}/ml/stt/transcribe",
            files={"file": f},
            data={"language_hint": source_lang},
            headers={"ngrok-skip-browser-warning": "true"}
        )

    original_text = stt_response.json()['text']
    print(f"Original ({source_lang}): {original_text}")

    # Step 2: Use provided translation (in real app, use translation API)
    print(f"Translated ({target_lang}): {target_text}")

    # Step 3: Synthesize
    tts_response = requests.post(
        f"{TTS_URL}/ml/tts/predict",
        json={"text": target_text, "language": target_lang},
        headers={
            "Content-Type": "application/json",
            "ngrok-skip-browser-warning": "true"
        }
    )

    result = tts_response.json()
    print(f"Audio generated: {result['duration']}s")

    return result

# Example: English to Hindi "translation"
translate_speech(
    "english_audio.wav",
    source_lang="en",
    target_lang="hi",
    target_text="नमस्ते, आपका स्वागत है।"  # Manually provided translation
)
```

---

## Performance Testing

### 1. Response Time Measurement

```python
import requests
import time
import statistics

STT_URL = "https://your-stt-url.ngrok-free.app"
TTS_URL = "https://your-tts-url.ngrok-free.app"

def measure_stt_latency(audio_file, num_runs=5):
    """Measure STT response times."""
    times = []

    for i in range(num_runs):
        with open(audio_file, "rb") as f:
            start = time.time()
            response = requests.post(
                f"{STT_URL}/ml/stt/transcribe",
                files={"file": f},
                headers={"ngrok-skip-browser-warning": "true"}
            )
            elapsed = time.time() - start
            times.append(elapsed)
            print(f"Run {i+1}: {elapsed:.2f}s")

    print(f"\nSTT Latency Stats:")
    print(f"  Min: {min(times):.2f}s")
    print(f"  Max: {max(times):.2f}s")
    print(f"  Mean: {statistics.mean(times):.2f}s")
    print(f"  Median: {statistics.median(times):.2f}s")

    return times

def measure_tts_latency(text, num_runs=5):
    """Measure TTS response times."""
    times = []

    for i in range(num_runs):
        start = time.time()
        response = requests.post(
            f"{TTS_URL}/ml/tts/predict",
            json={"text": text, "language": "en"},
            headers={
                "Content-Type": "application/json",
                "ngrok-skip-browser-warning": "true"
            }
        )
        elapsed = time.time() - start
        times.append(elapsed)
        print(f"Run {i+1}: {elapsed:.2f}s")

    print(f"\nTTS Latency Stats:")
    print(f"  Min: {min(times):.2f}s")
    print(f"  Max: {max(times):.2f}s")
    print(f"  Mean: {statistics.mean(times):.2f}s")
    print(f"  Median: {statistics.median(times):.2f}s")

    return times

# Run tests
print("STT Latency Test:")
stt_times = measure_stt_latency("test_audio.wav", num_runs=5)

print("\nTTS Latency Test:")
tts_times = measure_tts_latency("Hello, this is a test.", num_runs=5)
```

---

### 2. Concurrent Request Testing

**Warning:** Be careful with concurrent requests on Colab - GPU memory is limited!

```python
import requests
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

TTS_URL = "https://your-tts-url.ngrok-free.app"

def single_tts_request(text):
    """Make a single TTS request."""
    start = time.time()
    response = requests.post(
        f"{TTS_URL}/ml/tts/predict",
        json={"text": text, "language": "en"},
        headers={
            "Content-Type": "application/json",
            "ngrok-skip-browser-warning": "true"
        }
    )
    elapsed = time.time() - start
    return elapsed, response.status_code

def concurrent_test(num_requests=3, max_workers=2):
    """Test concurrent requests (limited to avoid OOM)."""
    texts = [f"This is test message number {i+1}." for i in range(num_requests)]

    print(f"Testing {num_requests} requests with {max_workers} workers...")
    start = time.time()

    results = []
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(single_tts_request, text): text for text in texts}

        for future in as_completed(futures):
            elapsed, status = future.result()
            results.append((elapsed, status))
            print(f"  Request completed: {elapsed:.2f}s, status={status}")

    total_time = time.time() - start

    print(f"\nConcurrent Test Results:")
    print(f"  Total time: {total_time:.2f}s")
    print(f"  Requests: {num_requests}")
    print(f"  Throughput: {num_requests/total_time:.2f} req/s")

    return results

# Run with caution!
concurrent_test(num_requests=3, max_workers=2)
```

---

## Error Handling

### Common Error Responses

```python
import requests

STT_URL = "https://your-stt-url.ngrok-free.app"
TTS_URL = "https://your-tts-url.ngrok-free.app"

def handle_stt_errors(file_path, language_hint=None):
    """Demonstrate error handling for STT."""
    try:
        with open(file_path, "rb") as f:
            response = requests.post(
                f"{STT_URL}/ml/stt/transcribe",
                files={"file": f},
                data={"language_hint": language_hint} if language_hint else {},
                headers={"ngrok-skip-browser-warning": "true"},
                timeout=120
            )

        if response.status_code == 200:
            return {"success": True, "data": response.json()}
        elif response.status_code == 400:
            return {"success": False, "error": "Bad request - check audio format"}
        elif response.status_code == 503:
            return {"success": False, "error": "Service unavailable - model not loaded"}
        else:
            return {"success": False, "error": f"HTTP {response.status_code}"}

    except requests.exceptions.Timeout:
        return {"success": False, "error": "Request timed out"}
    except requests.exceptions.ConnectionError:
        return {"success": False, "error": "Connection failed - check URL"}
    except FileNotFoundError:
        return {"success": False, "error": f"File not found: {file_path}"}

def handle_tts_errors(text, language="en"):
    """Demonstrate error handling for TTS."""
    try:
        if not text or len(text.strip()) == 0:
            return {"success": False, "error": "Text cannot be empty"}

        response = requests.post(
            f"{TTS_URL}/ml/tts/predict",
            json={"text": text, "language": language},
            headers={
                "Content-Type": "application/json",
                "ngrok-skip-browser-warning": "true"
            },
            timeout=120
        )

        if response.status_code == 200:
            return {"success": True, "data": response.json()}
        elif response.status_code == 422:
            return {"success": False, "error": "Validation error - check input"}
        elif response.status_code == 503:
            return {"success": False, "error": "Service unavailable"}
        else:
            return {"success": False, "error": f"HTTP {response.status_code}"}

    except requests.exceptions.Timeout:
        return {"success": False, "error": "Request timed out"}
    except requests.exceptions.ConnectionError:
        return {"success": False, "error": "Connection failed"}

# Test error handling
print(handle_stt_errors("nonexistent.wav"))
print(handle_tts_errors(""))
print(handle_tts_errors("Valid text", "en"))
```

---

## Multi-Language Testing

### Complete Language Test Suite

```python
import requests

STT_URL = "https://your-stt-url.ngrok-free.app"
TTS_URL = "https://your-tts-url.ngrok-free.app"

# Language test data
LANGUAGE_TESTS = {
    "en": {
        "text": "Hello, how are you today?",
        "expected_words": ["hello", "how", "are", "you"]
    },
    "hi": {
        "text": "नमस्ते, आप कैसे हैं?",
        "expected_words": ["नमस्ते", "आप", "कैसे"]
    },
    "es": {
        "text": "Hola, cómo estás?",
        "expected_words": ["hola", "cómo", "estás"]
    },
    "fr": {
        "text": "Bonjour, comment allez-vous?",
        "expected_words": ["bonjour", "comment"]
    },
    "de": {
        "text": "Guten Tag, wie geht es Ihnen?",
        "expected_words": ["guten", "tag", "wie"]
    },
    "ja": {
        "text": "こんにちは、お元気ですか？",
        "expected_words": ["こんにちは"]
    },
    "zh-cn": {
        "text": "你好，你今天怎么样？",
        "expected_words": ["你好"]
    }
}

def test_tts_language(lang, text):
    """Test TTS for a specific language."""
    response = requests.post(
        f"{TTS_URL}/ml/tts/predict",
        json={"text": text, "language": lang},
        headers={
            "Content-Type": "application/json",
            "ngrok-skip-browser-warning": "true"
        },
        timeout=60
    )

    if response.status_code == 200:
        result = response.json()
        return {
            "success": True,
            "duration": result.get("duration"),
            "status": result.get("status")
        }
    else:
        return {"success": False, "error": response.status_code}

def run_language_tests():
    """Run TTS tests for all supported languages."""
    print("=" * 60)
    print("MULTI-LANGUAGE TTS TEST SUITE")
    print("=" * 60)

    results = {}

    for lang, data in LANGUAGE_TESTS.items():
        print(f"\nTesting {lang.upper()}...")
        print(f"  Text: {data['text']}")

        result = test_tts_language(lang, data['text'])
        results[lang] = result

        if result['success']:
            print(f"  ✓ Success - Duration: {result['duration']}s")
        else:
            print(f"  ✗ Failed - Error: {result.get('error')}")

    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)

    success_count = sum(1 for r in results.values() if r['success'])
    print(f"Passed: {success_count}/{len(LANGUAGE_TESTS)}")

    return results

# Run the test suite
results = run_language_tests()
```

---

## Sample Audio Files

### Where to Get Test Audio

1. **LibriSpeech Dataset**: Free English audiobook readings
   - https://www.openslr.org/12/

2. **Mozilla Common Voice**: Multi-language audio
   - https://commonvoice.mozilla.org/

3. **Generate with TTS**: Use the TTS service to generate test audio

4. **Record Your Own**: Use your phone or computer microphone

### Generate Test Audio with TTS

```python
import requests

TTS_URL = "https://your-tts-url.ngrok-free.app"

def generate_test_audio(text, language, filename):
    """Generate test audio using TTS service."""
    response = requests.post(
        f"{TTS_URL}/ml/tts/predict",
        json={"text": text, "language": language},
        headers={
            "Content-Type": "application/json",
            "ngrok-skip-browser-warning": "true"
        }
    )

    if response.status_code == 200:
        result = response.json()
        print(f"Generated: {filename}")
        print(f"  Server path: {result['audio_path']}")
        print(f"  Duration: {result['duration']}s")
        return result
    else:
        print(f"Failed to generate {filename}")
        return None

# Generate test audio in multiple languages
test_phrases = [
    ("Hello, this is a test audio file.", "en", "test_english.wav"),
    ("नमस्ते, यह एक परीक्षण ऑडियो है।", "hi", "test_hindi.wav"),
    ("Bonjour, ceci est un fichier audio de test.", "fr", "test_french.wav"),
]

for text, lang, filename in test_phrases:
    generate_test_audio(text, lang, filename)
```

---

## Next Steps

1. **Run the Python test script**: `python test_services.py`
2. **Import Postman collection**: `postman_collection.json`
3. **Check troubleshooting guide**: `TROUBLESHOOTING.md`
4. **Monitor services**: `python monitor.py`

---

**Happy Testing!**
