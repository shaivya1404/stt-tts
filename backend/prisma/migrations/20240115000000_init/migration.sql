-- Enable UUID generation
CREATE EXTENSION IF NOT EXISTS "pgcrypto";

-- Create Enum Types
CREATE TYPE "UserRole" AS ENUM ('owner', 'admin', 'developer', 'viewer');
CREATE TYPE "VoiceProfileStatus" AS ENUM ('active', 'training', 'disabled');
CREATE TYPE "AudioFileType" AS ENUM ('tts_output', 'stt_input', 'stt_output', 'voice_clone_sample');
CREATE TYPE "JobStatus" AS ENUM ('queued', 'processing', 'completed', 'failed');
CREATE TYPE "MlModelType" AS ENUM ('tts', 'stt');
CREATE TYPE "MlModelStatus" AS ENUM ('active', 'deprecated', 'loading', 'error');
CREATE TYPE "UsageType" AS ENUM ('tts', 'stt');

-- organizations
CREATE TABLE "organizations" (
    "id" UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    "name" TEXT NOT NULL,
    "slug" TEXT NOT NULL UNIQUE,
    "created_at" TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    "updated_at" TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

-- users
CREATE TABLE "users" (
    "id" UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    "org_id" UUID NOT NULL REFERENCES "organizations"("id") ON DELETE CASCADE,
    "email" TEXT NOT NULL,
    "password_hash" TEXT NOT NULL,
    "role" "UserRole" NOT NULL DEFAULT 'developer',
    "created_at" TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    "updated_at" TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    UNIQUE ("org_id", "email")
);

-- api_keys
CREATE TABLE "api_keys" (
    "id" UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    "org_id" UUID NOT NULL REFERENCES "organizations"("id") ON DELETE CASCADE,
    "key_hash" TEXT NOT NULL UNIQUE,
    "name" TEXT NOT NULL,
    "scopes" TEXT[] NOT NULL DEFAULT ARRAY[]::TEXT[],
    "rate_limit_per_minute" INTEGER,
    "created_at" TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    "last_used_at" TIMESTAMPTZ,
    "revoked_at" TIMESTAMPTZ
);

-- voice_profiles
CREATE TABLE "voice_profiles" (
    "id" UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    "org_id" UUID NOT NULL REFERENCES "organizations"("id") ON DELETE CASCADE,
    "name" TEXT NOT NULL,
    "language" TEXT NOT NULL,
    "gender" TEXT,
    "description" TEXT,
    "base_model" TEXT NOT NULL,
    "speaker_embedding_path" TEXT,
    "status" "VoiceProfileStatus" NOT NULL DEFAULT 'training',
    "created_at" TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    "updated_at" TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

-- audio_files
CREATE TABLE "audio_files" (
    "id" UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    "org_id" UUID NOT NULL REFERENCES "organizations"("id") ON DELETE CASCADE,
    "user_id" UUID REFERENCES "users"("id") ON DELETE SET NULL,
    "type" "AudioFileType" NOT NULL,
    "storage_path" TEXT NOT NULL,
    "mime_type" TEXT NOT NULL,
    "duration_seconds" DOUBLE PRECISION,
    "size_bytes" BIGINT,
    "created_at" TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

-- tts_jobs
CREATE TABLE "tts_jobs" (
    "id" UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    "org_id" UUID NOT NULL REFERENCES "organizations"("id") ON DELETE CASCADE,
    "user_id" UUID REFERENCES "users"("id") ON DELETE SET NULL,
    "api_key_id" UUID REFERENCES "api_keys"("id") ON DELETE SET NULL,
    "input_text" TEXT NOT NULL,
    "language" TEXT NOT NULL,
    "voice_profile_id" UUID REFERENCES "voice_profiles"("id") ON DELETE SET NULL,
    "emotion" TEXT,
    "speed" DOUBLE PRECISION,
    "status" "JobStatus" NOT NULL DEFAULT 'queued',
    "result_audio_file_id" UUID REFERENCES "audio_files"("id") ON DELETE SET NULL,
    "error_message" TEXT,
    "created_at" TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    "updated_at" TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    "completed_at" TIMESTAMPTZ
);

-- stt_jobs
CREATE TABLE "stt_jobs" (
    "id" UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    "org_id" UUID NOT NULL REFERENCES "organizations"("id") ON DELETE CASCADE,
    "user_id" UUID REFERENCES "users"("id") ON DELETE SET NULL,
    "api_key_id" UUID REFERENCES "api_keys"("id") ON DELETE SET NULL,
    "input_audio_file_id" UUID NOT NULL REFERENCES "audio_files"("id") ON DELETE CASCADE,
    "language_detected" TEXT,
    "model_used" TEXT NOT NULL,
    "status" "JobStatus" NOT NULL DEFAULT 'queued',
    "error_message" TEXT,
    "created_at" TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    "updated_at" TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    "completed_at" TIMESTAMPTZ
);

-- transcriptions
CREATE TABLE "transcriptions" (
    "id" UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    "stt_job_id" UUID NOT NULL UNIQUE REFERENCES "stt_jobs"("id") ON DELETE CASCADE,
    "text" TEXT NOT NULL,
    "language" TEXT NOT NULL,
    "confidence" DOUBLE PRECISION NOT NULL,
    "timestamps" JSONB NOT NULL,
    "created_at" TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

-- usage_records
CREATE TABLE "usage_records" (
    "id" UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    "org_id" UUID NOT NULL REFERENCES "organizations"("id") ON DELETE CASCADE,
    "api_key_id" UUID REFERENCES "api_keys"("id") ON DELETE SET NULL,
    "type" "UsageType" NOT NULL,
    "units" DOUBLE PRECISION NOT NULL,
    "metadata" JSONB,
    "created_at" TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

-- ml_models
CREATE TABLE "ml_models" (
    "id" UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    "name" TEXT NOT NULL,
    "type" "MlModelType" NOT NULL,
    "version" TEXT NOT NULL,
    "status" "MlModelStatus" NOT NULL DEFAULT 'loading',
    "config" JSONB,
    "created_at" TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    "updated_at" TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    UNIQUE ("name", "type", "version")
);

-- trigger function for updated_at columns
CREATE OR REPLACE FUNCTION set_updated_at()
RETURNS TRIGGER AS $$
BEGIN
  NEW."updated_at" = NOW();
  RETURN NEW;
END;
$$ LANGUAGE plpgsql;

CREATE TRIGGER organizations_updated_at
BEFORE UPDATE ON "organizations"
FOR EACH ROW EXECUTE PROCEDURE set_updated_at();

CREATE TRIGGER users_updated_at
BEFORE UPDATE ON "users"
FOR EACH ROW EXECUTE PROCEDURE set_updated_at();

CREATE TRIGGER voice_profiles_updated_at
BEFORE UPDATE ON "voice_profiles"
FOR EACH ROW EXECUTE PROCEDURE set_updated_at();

CREATE TRIGGER tts_jobs_updated_at
BEFORE UPDATE ON "tts_jobs"
FOR EACH ROW EXECUTE PROCEDURE set_updated_at();

CREATE TRIGGER stt_jobs_updated_at
BEFORE UPDATE ON "stt_jobs"
FOR EACH ROW EXECUTE PROCEDURE set_updated_at();

CREATE TRIGGER ml_models_updated_at
BEFORE UPDATE ON "ml_models"
FOR EACH ROW EXECUTE PROCEDURE set_updated_at();
