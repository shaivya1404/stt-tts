# STT & TTS ML Services — Backend Integration Guide

> **For**: Backend developer replacing old STT/TTS
> **Architecture**: Your Node.js backend (port 4000) calls Python ML services via HTTP
> **No changes needed in ML services** — only replace calls in the backend

---

## Architecture Overview

```
Your Backend (Node.js :4000)
    │
    ├──HTTP POST──→  STT Python Service (:8002)   ← Whisper model
    │                  /ml/stt/transcribe
    │
    └──HTTP POST──→  TTS Python Service (:8001)   ← XTTS v2 model
                       /ml/tts/predict
```

---

## 1. Environment Variables

Add these to your `.env`:

```env
TTS_SERVICE_URL=http://localhost:8001      # or http://tts-service:8001 in Docker
STT_SERVICE_URL=http://localhost:8002      # or http://stt-service:8002 in Docker
```

---

## 2. STT (Speech-to-Text) — ML Service API

### `POST {STT_SERVICE_URL}/ml/stt/transcribe`

Sends audio, gets back transcribed text.

**Request**: `multipart/form-data`

| Field           | Type   | Required | Description                              |
|-----------------|--------|----------|------------------------------------------|
| `file`          | binary | YES      | Audio file (WAV, MP3, FLAC, OGG, WEBM)  |
| `language_hint` | string | no       | `"hi"`, `"en"`, `"ta"`, `"te"`, etc.    |

**Max file size**: 30 MB

**cURL Example**:
```bash
curl -X POST http://localhost:8002/ml/stt/transcribe \
  -F "file=@recording.wav" \
  -F "language_hint=hi"
```

**Response** (`200 OK`):
```json
{
  "text": "नमस्ते यह एक परीक्षण है",
  "language": "hi",
  "confidence": 0.92,
  "timestamps": [
    { "start": 0.0, "end": 0.8, "word": "नमस्ते" },
    { "start": 0.9, "end": 1.2, "word": "यह" },
    { "start": 1.3, "end": 1.6, "word": "एक" },
    { "start": 1.7, "end": 2.3, "word": "परीक्षण" },
    { "start": 2.4, "end": 2.7, "word": "है" }
  ],
  "meta": {
    "duration_seconds": 2.7,
    "quality_score": 0.85,
    "language_detected": "hi",
    "model_name": "whisper_large-v3",
    "audio_samples": 43200
  },
  "modelUsed": "whisper_large-v3:faster-whisper",
  "status": "success"
}
```

**Response Fields**:

| Field        | Type     | Description                                       |
|--------------|----------|---------------------------------------------------|
| `text`       | string   | Transcribed text                                  |
| `language`   | string   | Detected language code                            |
| `confidence` | float    | 0.0 – 1.0 confidence score                       |
| `timestamps` | array    | Word-level timestamps `{start, end, word}`        |
| `meta`       | object   | Extra info: duration, quality score, model name   |
| `modelUsed`  | string   | Which model was used                              |
| `status`     | string   | `"success"` or `"error"`                          |

### `GET {STT_SERVICE_URL}/ml/stt/health`

**Response**:
```json
{
  "status": "healthy",
  "models": [
    { "type": "stt", "name": "whisper_large-v3", "version": "v1", "status": "ready" }
  ],
  "device": "cuda"
}
```

---

## 3. TTS (Text-to-Speech) — ML Service API

### `POST {TTS_SERVICE_URL}/ml/tts/predict`

Sends text, gets back audio.

**Request**: `application/json`

```json
{
  "text": "नमस्ते, यह एक परीक्षण है",
  "language": "hi",
  "voice_id": "default",
  "speaker_wav": null,
  "speed": 1.0,
  "emotion": null
}
```

| Field         | Type   | Required | Default     | Description                                     |
|---------------|--------|----------|-------------|-------------------------------------------------|
| `text`        | string | YES      | —           | Text to speak                                   |
| `language`    | string | YES      | —           | Language code (see table below)                 |
| `voice_id`    | string | no       | `"default"` | Voice profile ID                                |
| `speaker_wav` | string | no       | `null`      | Path to reference WAV for voice cloning         |
| `speed`       | float  | no       | `1.0`       | Speed multiplier (0.5 – 2.0)                   |
| `emotion`     | string | no       | `null`      | Emotion style (future feature)                  |

**cURL Example**:
```bash
curl -X POST http://localhost:8001/ml/tts/predict \
  -H "Content-Type: application/json" \
  -d '{
    "text": "नमस्ते, यह एक परीक्षण है",
    "language": "hi",
    "voice_id": "default",
    "speed": 1.0
  }'
```

**Response** (`200 OK`):
```json
{
  "audio_path": "/models/tts/xtts_v2/v1/synthesized/default_abc123.wav",
  "audio_url": "/ml/tts/audio/default_abc123.wav",
  "audio_base64": "UklGRi4gAABXQVZFZm10...",
  "duration": 3.2,
  "status": "success",
  "meta": {
    "language": "hi",
    "voice_id": "default",
    "speed": 1.0,
    "mos_score": 3.8,
    "model": "xtts_v2",
    "version": "v1",
    "char_count": 27
  }
}
```

**Response Fields**:

| Field          | Type   | Description                                           |
|----------------|--------|-------------------------------------------------------|
| `audio_path`   | string | Server-side file path of generated WAV                |
| `audio_url`    | string | Relative URL to download the audio                    |
| `audio_base64` | string | Base64-encoded WAV audio (use this to return to user) |
| `duration`     | float  | Audio duration in seconds                             |
| `status`       | string | `"success"` or `"error"`                              |
| `meta`         | object | Quality score, char count, model info                 |

### `GET {TTS_SERVICE_URL}/ml/tts/audio/{filename}`

Download a generated audio file directly.

**Example**: `GET http://localhost:8001/ml/tts/audio/default_abc123.wav`
**Response**: Binary WAV file (`Content-Type: audio/wav`)

### `GET {TTS_SERVICE_URL}/ml/tts/health`

**Response**:
```json
{
  "status": "healthy",
  "models": [
    { "type": "tts", "name": "xtts_v2", "version": "v1", "status": "ready" }
  ],
  "device": "cuda"
}
```

---

## 4. Supported Languages

| Code    | Language | STT | TTS |
|---------|----------|-----|-----|
| `hi`    | Hindi    | YES | YES |
| `en`    | English  | YES | YES |
| `ta`    | Tamil    | YES | YES |
| `te`    | Telugu   | YES | YES |
| `bn`    | Bengali  | YES | YES |
| `es`    | Spanish  | YES | YES |
| `fr`    | French   | YES | YES |
| `de`    | German   | YES | YES |
| `ar`    | Arabic   | YES | YES |
| `zh-cn` | Chinese  | YES | YES |
| `ja`    | Japanese | YES | YES |
| `ko`    | Korean   | YES | YES |

STT supports 99 languages total (Whisper). TTS supports 17 languages (XTTS v2).

---

## 5. Existing Backend Integration Code (Reference)

The Node.js backend already has working ML client code. Use these as reference:

### STT Client — `backend/src/services/mlSttClient.service.ts`

```typescript
// How the backend calls STT service
const formData = new FormData();
formData.append('file', audioBuffer, { filename, contentType: mimeType });
if (languageHint) formData.append('language_hint', languageHint);

const response = await axios.post(
  `${STT_SERVICE_URL}/ml/stt/transcribe`,
  formData,
  { headers: formData.getHeaders(), maxBodyLength: Infinity }
);

// response.data → { text, language, confidence, timestamps, meta, modelUsed }
```

### TTS Client — `backend/src/services/mlTtsClient.service.ts`

```typescript
// How the backend calls TTS service
const response = await axios.post(`${TTS_SERVICE_URL}/ml/tts/predict`, {
  text: "Hello world",
  language: "en",
  voice_id: "default",
  speed: 1.0,
});

// response.data → { audio_path, audio_url, audio_base64, duration, status, meta }
```

---

## 6. Backend API Routes (What Frontend/App Calls)

These are the existing routes on the Node.js backend (port 4000):

### STT Routes

| Method | Route                              | Body                          | Response                                              |
|--------|------------------------------------|-------------------------------|-------------------------------------------------------|
| POST   | `/api/v1/stt/transcribe`           | `multipart: audio_file` + query `?language_hint=hi` | `{ job_id, text, language, confidence, timestamps }` |
| POST   | `/api/v1/stt/batch-transcribe`     | `multipart: audio_files[]`    | `{ items: [{ job_id, text, language, confidence, timestamps }] }` |
| POST   | `/api/v1/stt/transcribe-realtime`  | —                             | `501 Not Implemented`                                 |

### TTS Routes

| Method | Route                          | Body                                                        | Response                                          |
|--------|--------------------------------|-------------------------------------------------------------|---------------------------------------------------|
| POST   | `/api/v1/tts/synthesize`       | `{ text, language, voice_id?, emotion?, speed? }`          | `{ job_id, audio_url, duration, status }`         |
| POST   | `/api/v1/tts/synthesize-batch` | `{ items: [{ text, language, voice_id?, emotion?, speed? }] }` | `{ items: [{ job_id, audio_url, duration, status }] }` |
| GET    | `/api/v1/tts/voices`           | —                                                           | `[{ id, name, language, gender, status }]`        |
| POST   | `/api/v1/tts/voice-clone`      | `multipart: audio_sample` + `{ name, language, gender?, description? }` | `{ id, name, language, gender, status }` |

**Auth**: All routes support JWT (`Authorization: Bearer <token>`) or API key (`X-API-Key: <key>`).

---

## 7. Error Handling

Both ML services return errors as:

```json
{
  "detail": "Error message here"
}
```

HTTP status codes:
- `200` — Success
- `400` — Bad request (missing file, invalid input)
- `422` — Validation error
- `500` — ML service internal error (model crash, OOM, etc.)

**Important**: If the ML service is down, your axios/fetch call will throw a connection error. Wrap calls in try/catch.

---

## 8. Running Everything

### Option A: Docker (Recommended)

```bash
cd tts-stt/infra/docker
docker-compose -f docker-compose.dev.yml up
```

This starts: postgres (5432), redis (6379), tts-service (8001), stt-service (8002), backend (4000), frontend (5173)

### Option B: Manual

```bash
# Terminal 1 — STT Service
cd tts-stt/ml-service/stt-service
pip install -r requirements.txt
uvicorn app:app --host 0.0.0.0 --port 8002

# Terminal 2 — TTS Service
cd tts-stt/ml-service/tts-service
pip install -r requirements.txt
uvicorn app:app --host 0.0.0.0 --port 8001

# Terminal 3 — Backend
cd tts-stt/backend
npm install
npm run dev
```

### Health Check

```bash
curl http://localhost:8001/ml/tts/health   # TTS
curl http://localhost:8002/ml/stt/health   # STT
curl http://localhost:4000/health           # Backend
```

---

## 9. What to Replace

If you had an old STT/TTS provider (Google Cloud, Azure, etc.), replace:

1. **Old STT call** → `POST {STT_SERVICE_URL}/ml/stt/transcribe` (multipart with `file` field)
2. **Old TTS call** → `POST {TTS_SERVICE_URL}/ml/tts/predict` (JSON with `text` + `language`)
3. **Old config/API keys** → Replace with `STT_SERVICE_URL` and `TTS_SERVICE_URL` env vars
4. **Old response parsing** → Use the response formats documented above

The key files to modify in the backend are:
- `backend/src/services/mlSttClient.service.ts` — STT HTTP client (already done)
- `backend/src/services/mlTtsClient.service.ts` — TTS HTTP client (already done)
- `backend/src/config/index.ts` — Service URLs (already configured)

---

## 10. Models Info

| Model | Type | Size | Languages | Notes |
|-------|------|------|-----------|-------|
| Faster-Whisper large-v3 | STT | 1.5B params | 99 languages | Production model, auto-downloads |
| Whisper Small + LoRA | STT | 244M + 7MB | Hindi, English | Fine-tuned model in `whisper-stt-model-29-1-2026/` |
| XTTS v2 (Coqui) | TTS | ~1.5GB | 17 languages | Voice cloning, auto-downloads |

**First startup will be slow** — models download automatically (~3-5 GB total). Subsequent starts are fast.

**GPU**: CUDA GPU recommended. Falls back to CPU (much slower).
