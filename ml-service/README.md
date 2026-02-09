# ML Service (TTS & STT)

Phase 3 introduces production-ready scaffolding for the text-to-speech (TTS) and speech-to-text (STT) microservices that power the monorepo. Each service now exposes modular inference pipelines, an in-memory model registry, and hot-reload hooks so the backend can coordinate deployments.

## Shared Infrastructure

- **Configuration** — `common/config.py` exposes a `Settings` object (via Pydantic BaseSettings) that reads environment variables such as `MODEL_BASE_PATH`, `LOG_LEVEL`, and `DEVICE`.
- **Logging** — `common/logging_config.py` configures Loguru with ISO timestamps and request-id placeholders.
- **Model Registry** — `common/registry.py` provides an in-memory registry used by both services to advertise available models and to simulate hot-reloads.

Environment variables:

| Variable | Default | Description |
| --- | --- | --- |
| `MODEL_BASE_PATH` | `/models` | Root location where model artifacts and synthesized assets are stored. |
| `LOG_LEVEL` | `INFO` | Log level passed to Loguru. |
| `DEVICE` | auto-detected | Forces `cpu`/`cuda` selection; otherwise auto-detected (if torch is available). |

## TTS Service

- Location: `ml-service/tts-service`
- Pipeline entrypoint: `core/pipeline.py` (`TTSPipeline`)
- Placeholder components: language ID, text normalization, G2P, speaker encoder, style/emotion conditioning, VITS acoustic model wrapper, HiFi-GAN vocoder, audio post-processing, MOSNet quality estimation.
- Default model registered as `vits_multispkr_indic:v1`.

### REST Endpoints

| Method | Path | Description |
| --- | --- | --- |
| `GET` | `/ml/tts/health` | Service health plus registry snapshot. |
| `GET` | `/ml/tts/models` | List registered TTS models. |
| `POST` | `/ml/tts/initialize` | Rebuild pipelines (used during bootstrapping). |
| `POST` | `/ml/tts/reload` | Simulate hot-reloading by toggling registry status. |
| `POST` | `/ml/tts/predict` | Run the full TTS pipeline (text → audio). |

### Running Locally

```bash
cd ml-service/tts-service
uvicorn app:app --host 0.0.0.0 --port 8001
```

### Docker

```
docker build -f infra/docker/Dockerfile.tts -t tts-service .
docker run --rm -p 8001:8001 tts-service
```

Mount `${MODEL_BASE_PATH}` if you need to persist synthesized audio or host real model checkpoints.

## STT Service

- Location: `ml-service/stt-service`
- Pipeline entrypoint: `core/pipeline.py` (`STTPipeline`)
- Placeholder components: audio preprocessing, RNNoise, AEC, VAD, language ID, Conformer-RNNT primary ASR, Whisper fallback, language model refinement, punctuation, truecasing, ITN, quality scoring.
- Default model registered as `conformer_rnnt_indic:v1`.
- `/ml/stt/stream` currently returns a stubbed 501 response; real-time streaming will arrive in a later phase.

### REST Endpoints

| Method | Path | Description |
| --- | --- | --- |
| `GET` | `/ml/stt/health` | Service health plus registry snapshot. |
| `GET` | `/ml/stt/models` | List registered STT models. |
| `POST` | `/ml/stt/initialize` | Rebuild pipelines / reload models. |
| `POST` | `/ml/stt/reload` | Simulate hot-reload cycle. |
| `POST` | `/ml/stt/transcribe` | Accepts an audio file upload and returns structured transcription data. |
| `POST` | `/ml/stt/stream` | Stub endpoint for future streaming transcription. |

### Running Locally

```bash
cd ml-service/stt-service
uvicorn app:app --host 0.0.0.0 --port 8002
```

### Docker

```
docker build -f infra/docker/Dockerfile.stt -t stt-service .
docker run --rm -p 8002:8002 stt-service
```

## Where to Plug in Real Models

- Replace placeholder logic inside each `core/` component. For example, drop the real language-ID model into `language_id.py`, connect your fine-tuned VITS stack under `vits_wrapper.py`, wire Conformer checkpoints into `asr_conformer_rnnt.py`, etc.
- Store checkpoints beneath `${MODEL_BASE_PATH}` (default `/models`) so both Docker and host deployments share a consistent layout.
- Use the `/ml/*/reload` endpoints to hook hot-reload logic once actual models are ready.

## docker-compose

`infra/docker/docker-compose.dev.yml` already references these services. After running `docker compose -f infra/docker/docker-compose.dev.yml up tts stt`, both endpoints will be exposed on ports `8001` and `8002` respectively.
