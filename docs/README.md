# TTS/STT Platform Monorepo

## Overview
The repository now provides the foundational scaffold for an enterprise-ready Text-to-Speech (TTS) and Speech-to-Text (STT) platform. It is organized as a monorepo so that backend services, machine learning microservices, frontend applications, infrastructure tooling, and documentation can evolve together in a consistent manner.

## High-Level Architecture
- **Backend API (Node.js + Express + TypeScript):** Serves as the main entry point for clients and other systems. It surfaces health and version endpoints today, and will host future orchestration logic connecting storage, queues, billing, and ML pipelines.
- **ML Services (Python + FastAPI):** Dedicated microservices for TTS and STT workloads. Each exposes basic health, initialization, and inference-style endpoints that will later connect to actual models.
- **Frontend (React + Vite + TypeScript):** A lightweight dashboard that validates end-to-end connectivity and provides a base for future operator and customer-facing experiences.
- **Infra (Docker, Docker Compose, nginx, k8s stubs coming later):** Container definitions for every service plus supporting databases and caches. Local developers can run everything via a single compose file.

## Repository Layout
```
backend/     → Node.js + Express + TypeScript API service
ml-service/  → Python FastAPI microservices for TTS & STT with shared config stubs
frontend/    → React + Vite + TypeScript dashboard
infra/       → Dockerfiles, docker-compose stack, nginx config, and Kubernetes manifest stubs
docs/        → Platform documentation & onboarding guides
first        → Legacy placeholder file retained for history
```

## Running Services Locally (without Docker)
### Backend API
```bash
cd backend
npm install
npm run dev
# Server listens on http://localhost:4000 by default
```

### TTS Service
```bash
cd ml-service/tts-service
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
uvicorn app:app --reload --port 8001
```

### STT Service
```bash
cd ml-service/stt-service
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
uvicorn app:app --reload --port 8002
```

### Frontend Dashboard
```bash
cd frontend
npm install
npm run dev
# Vite dev server runs on http://localhost:5173
```

Use the `.env.example` files inside `backend/` and `frontend/` as references for local configuration.

## Database & Prisma (Phase 2)
Phase 2 introduces the full persistence layer via Prisma + PostgreSQL. All schema definitions live in `backend/prisma/schema.prisma` and the generated SQL migration is committed under `backend/prisma/migrations/`.

Common commands:
```bash
cd backend
npm install
# Generate Prisma client
npm run prisma:generate
# Apply migrations to your local DB
npm run prisma:migrate
# Seed a default org + owner account
npm run prisma:seed
```
The seed step will create an `Acme Corp` organization with an owner account (`owner@acme.dev` / `ChangeMe123!`) and two baseline ML model rows so status endpoints return data. Update the corresponding `SEED_*` environment variables if you want to customize credentials.

## Auth & Security
- **JWT login** – `POST /api/v1/auth/login` accepts email/password (and an optional `orgSlug` if the same email exists in multiple orgs) and returns a Bearer token. Include it via `Authorization: Bearer <token>` for any dashboard/admin route.
- **API keys** – Owners/Admins can manage keys under `/api/v1/auth/api-keys`. Keys are shown once at creation time, hashed in the DB, and can be scoped to `tts` and/or `stt`. Supply them via the `x-api-key` header when calling job endpoints server-to-server.
- **RBAC** – Middleware enforces roles for sensitive operations (e.g., model reloads, API key management, voice cloning). Rate limiting is applied per org/API key with configurable windows in `.env`.
- **Dev-mode public inference endpoints** – During local development the TTS/STT/model/analytics routes are mounted without `requireAuthOrApiKey`, so you can hit them from curl/Postman without headers. When a JWT or API key is provided the normal scope/rate-limit enforcement still applies.

Anonymous calls are associated with the organization identified by `DEV_ORG_ID` (optionally paired with `DEV_ORG_SLUG`/`DEV_ORG_NAME`). If those variables are not set, the backend reuses the first organization in the database or creates a placeholder "Development Playground" org so usage records continue to work.

## REST API Endpoints (Phase 2)
All endpoints share the `/api/v1` prefix. Full OpenAPI docs are hosted at `http://localhost:4000/api/v1/docs` once the backend is running.

### TTS
| Method | Path | Auth | Description |
| --- | --- | --- | --- |
| POST | `/tts/synthesize` | Public (JWT/API key optional) | Synchronously synthesize text to speech. Returns job metadata plus audio URL. API keys still need the `tts` scope when supplied.
| POST | `/tts/synthesize-batch` | Public (JWT/API key optional) | Submit multiple synthesis requests in one payload; optional API keys still require the `tts` scope.
| GET | `/tts/voices` | Public (JWT/API key optional) | List the organization’s voice profiles for the active org.
| POST | `/tts/voice-clone` | Public (JWT/API key optional) | Upload a sample (`audio_sample`) to create a training voice profile. Audio is stored via the S3/MinIO adapter and the profile is marked `training`.

Auth headers remain optional in dev mode—add `Authorization` or `x-api-key` only when you want to test those flows.

Sample cURL:
```bash
curl -X POST http://localhost:4000/api/v1/tts/synthesize \
  -H "Authorization: Bearer <TOKEN>" \
  -H "Content-Type: application/json" \
  -d '{"text": "Hello world", "language": "en-US"}'
```

**Batch synthesis**
```bash
curl -X POST http://localhost:4000/api/v1/tts/synthesize-batch \
  -H "Authorization: Bearer <TOKEN>" \
  -H "Content-Type: application/json" \
  -d '{"items": [
        {"text": "Hello team", "language": "en-IN", "voice_id": "default"},
        {"text": "Namaste Duniya", "language": "hi-IN", "emotion": "happy"}
      ]}'
```

**List voices**
```bash
curl -X GET http://localhost:4000/api/v1/tts/voices \
  -H "Authorization: Bearer <TOKEN>"
```

**Voice clone upload**
```bash
curl -X POST http://localhost:4000/api/v1/tts/voice-clone \
  -H "Authorization: Bearer <TOKEN>" \
  -F "audio_sample=@/path/to/sample.wav" \
  -F "name=Indic Voice" \
  -F "language=hi-IN" \
  -F "description=Customer provided clone"
```

### End-to-end TTS Flow
1. **Request intake** – when auth headers are present the backend validates JWT/API key scopes, enforces rate limits, and normalizes payloads using the Zod schemas defined in `src/controllers/tts.controller.ts`.
2. **Job tracking** – a `tts_jobs` row is inserted with status `processing`, linked to the organization, optional voice profile, and the original text/emotion/speed metadata.
3. **ML invocation** – `TtsService` bridges to the FastAPI microservice at `/ml/tts/predict`. The ML service now runs a modular pipeline (language ID → text normalization → G2P → ECAPA-TDNN speaker encoder → style/emotion conditioning → VITS → HiFi-GAN → RNNoise → MOSNet) and returns `{ audio_path, duration, status, meta }`.
4. **Persistence & billing** – successful runs create an `audio_files` record pointing at the ML-provided `audio_path`, mark the job `completed`, and emit a `usage_records` row measuring either `meta.char_count` or the raw character count. Failures set the job status to `failed` with the propagated error.

Voice cloning uploads the `audio_sample` to `voice-clone/<orgId>/...`, stores it as an `audio_files` row of type `voice_clone_sample`, and creates a `voice_profiles` entry with status `training` plus a placeholder `speaker_embedding_path` for downstream fine-tuning.

### ML TTS Microservice
| Method | Path | Description |
| --- | --- | --- |
| `GET` | `/ml/tts/health` | Returns overall service status plus the in-memory model registry. |
| `GET` | `/ml/tts/models` | Lists registered TTS models (`vits_multispkr_indic:v1` by default). |
| `POST` | `/ml/tts/initialize` | Rebuilds and warms the in-process pipeline, marking the registry entry as `ready`. |
| `POST` | `/ml/tts/reload` | Simulates a hot-reload cycle (set to `loading`, sleep, reinitialize). |
| `POST` | `/ml/tts/predict` | Accepts `application/json` `{ text, language?, voice_id?, emotion?, speed? }` and runs the modular pipeline, returning `{ audio_path, duration, status, meta }`. |

Sample `curl` hitting `/ml/tts/predict` directly:
```bash
curl -X POST http://localhost:8001/ml/tts/predict \
  -H "Content-Type: application/json" \
  -d '{
        "text": "Team update at 5pm",
        "language": "en-IN",
        "emotion": "calm",
        "speed": 1.1
      }'
```

The `meta` payload exposes diagnostics such as the detected language, phoneme count, MOS estimate, character count, and the model/version pair so downstream services can log or bill accurately.

### STT
| Method | Path | Auth | Description |
| --- | --- | --- | --- |
| POST | `/stt/transcribe` | Public (JWT/API key optional) | Upload a single audio file (`audio_file`) for transcription. API keys still need the `stt` scope when supplied.
| POST | `/stt/batch-transcribe` | Public (JWT/API key optional) | Send multiple files (`audio_files`) and receive per-job IDs.
| POST | `/stt/transcribe-realtime` | Public (JWT/API key optional) | REST stub returning `501` until the realtime WebSocket endpoint ships in Phase 5.

**Single-file transcription**

- Endpoint: `POST /api/v1/stt/transcribe?language_hint=<optional-lang>`
- Body: `multipart/form-data` with the field `audio_file` (max 30 MB). Auth headers are optional in dev mode; when you include an API key it still needs the `stt` scope.
- Response shape: `{ job_id, text, language, confidence, timestamps }`. Each call creates an `audio_files` row, kicks off an `stt_job`, stores the transcription payload, and records an `usage_records` entry with the detected duration.

Example request:

```bash
curl -X POST "http://localhost:4000/api/v1/stt/transcribe?language_hint=en-IN" \
  -F "audio_file=@/path/to/sample.wav"
```

Add `-H "Authorization: Bearer <TOKEN>"` or `-H "x-api-key: <KEY>"` when you want to test authenticated flows.

**Batch transcription**

- Endpoint: `POST /api/v1/stt/batch-transcribe`
- Body: `multipart/form-data` with repeated `audio_files` parts. Returns `{ "items": [ ... ] }`, where every element mirrors the single-file response payload for the corresponding upload.

### End-to-end STT Flow
1. A client hits either `/api/v1/stt/transcribe` or `/api/v1/stt/batch-transcribe`. When auth headers are present the backend validates JWT/API key scopes, enforces rate limits, and uploads the incoming audio to the `audio_files` table via the storage helper (S3/MinIO compatible).
2. A corresponding row is inserted in `stt_jobs` (status `processing`). The backend then hands the raw buffer off to `mlSttClient`, which bridges to the FastAPI service at `/ml/stt/transcribe`.
3. The STT FastAPI service runs the modular `STTPipeline` (Conformer RNNT primary + Whisper fallback, language ID, punctuation/ITN) and returns `{ text, language, confidence, timestamps, meta, modelUsed }`.
4. The backend persists the `transcriptions` row, updates the `stt_jobs` status, and records a `usage_records` entry measured in seconds so analytics stay in sync. The final REST response simply echoes the job metadata plus transcription payload.

### ML Model Management
| Method | Path | Auth | Description |
| --- | --- | --- | --- |
| GET | `/models/status` | Public (JWT/API key optional) | Returns all ML models plus health from the FastAPI services.
| POST | `/models/reload` | Public (JWT/API key optional) | Calls `initialize` on both ML services and updates model rows.

### ML STT Microservice
The FastAPI-based STT microservice (default `http://localhost:8002`) exposes the following endpoints, which the backend now calls via the new `mlSttClient` integration:

| Method | Path | Description |
| --- | --- | --- |
| `GET` | `/ml/stt/health` | Returns registry snapshot + readiness info. |
| `GET` | `/ml/stt/models` | Lists registered STT models and their versions. |
| `POST` | `/ml/stt/initialize` | Rebuilds the in-process pipeline (used at boot). |
| `POST` | `/ml/stt/reload` | Simulates a hot-reload cycle for active STT models. |
| `POST` | `/ml/stt/transcribe` | Accepts `multipart/form-data` (`file` + optional `language_hint` form fields) and runs the placeholder pipeline. |
| `POST` | `/ml/stt/stream` | Stub that currently returns `501`.

Sample `curl` hitting the ML service directly:

```bash
curl -X POST http://localhost:8002/ml/stt/transcribe \
  -F "file=@/path/to/sample.wav" \
  -F "language_hint=hi-IN"
```

The pipeline defined in `ml-service/stt-service/core/pipeline.py` chains audio preprocessing, denoising, AEC, VAD, language ID, Conformer RNNT inference, Whisper fallback, LM refinement, punctuation, truecasing, ITN, and quality scoring. It emits `meta.duration_seconds`, `timestamps`, and the `modelUsed` identifier so that the backend can store the transcription plus usage metrics.

### Analytics
| Method | Path | Auth | Description |
| --- | --- | --- | --- |
| GET | `/analytics/usage` | Public (JWT/API key optional) | Aggregated usage (total TTS chars, STT seconds, daily rollups for the last 30 days).

Swagger definitions describe request/response shapes; refer to them for full field-level documentation.

## TTS Training Workflows
TTS checkpoints live under `${MODEL_BASE_PATH}/tts/<model_name>/<version>/`. The new training stubs demonstrate how to produce multi-speaker acoustic and vocoder artifacts that the serving pipeline (`vits_wrapper`, `vocoder_hifigan`) will eventually load from disk.

1. **VITS multi-speaker skeleton**
   ```bash
   python ml-service/training/tts/train_vits.py \
     --config ml-service/training/tts/configs/vits_multispkr_indic_v1.yaml \
     --device cpu
   ```
   - Parses the shared YAML config (languages, manifests, optimizer, output layout), generates synthetic batches when manifests are missing, and saves checkpoints under `tts/<model_name>/<version>/checkpoints/` plus a `final.pt` snapshot.

2. **HiFi-GAN placeholder**
   ```bash
   python ml-service/training/tts/train_hifigan.py \
     --config ml-service/training/tts/configs/vits_multispkr_indic_v1.yaml \
     --device cuda
   ```
   - Trains a minimal MLP vocoder against dummy mel/waveform pairs and drops weights in `tts/<model_name>/<version>/hifigan/`. Use the same config so acoustic/vocoder checkpoints remain in sync.

Update `MODEL_BASE_PATH` (env var for the ML services) to point at a shared volume to make these files accessible to the FastAPI runtime.

## STT Training Workflows
STT checkpoints are stored under `${MODEL_BASE_PATH}/stt/<family>/<model_name>/<version>/`. The ML service loads artifacts from that layout during startup.

Two lightweight training entrypoints live in `ml-service/training/stt/`:

1. **Conformer RNNT placeholder**
   ```bash
   python ml-service/training/stt/train_conformer_rnnt.py \
     --config ml-service/training/stt/configs/conformer_indic_v1.yaml \
     --device cpu
   ```
   - Consumes JSONL manifests declared in the config, synthesizes random batches, and saves checkpoints to `stt/<model>/<version>/`.

2. **Whisper fine-tune skeleton**
   ```bash
   python ml-service/training/stt/train_whisper.py \
     --config ml-service/training/stt/configs/whisper_finetune_indic_v1.yaml \
     --device cuda
   ```
   - Demonstrates how to wire Hugging Face `datasets` + `transformers` with `language_hint` aware configs. It saves both the model and processor beside the checkpoints directory, ready for the serving pipeline to pick up.

Both scripts simply stub the training loops today—swap in real datasets/models and update the configs when you plug in production checkpoints.

## Running the Full Stack with Docker Compose
```bash
docker-compose -f infra/docker/docker-compose.dev.yml up --build
```
This command launches PostgreSQL, Redis, both ML services, the backend API, and the frontend dashboard. Once the containers are running you can check:
- Backend health: http://localhost:4000/health
- TTS service health: http://localhost:8001/ml/tts/health
- STT service health: http://localhost:8002/ml/stt/health
- Frontend dashboard: http://localhost:5173

The compose file wires service discovery (e.g., backend ↔ ML services) via container names, so cross-service health checks will work out of the box. Stop the stack with `Ctrl+C` or `docker-compose … down`.
