import dotenv from 'dotenv';

dotenv.config();

interface ServiceConfig {
  ttsServiceUrl: string;
  sttServiceUrl: string;
}

interface PostgresConfig {
  host: string;
  port: number;
  user: string;
  password: string;
  database: string;
}

interface RedisConfig {
  host: string;
  port: number;
  password?: string;
}

interface JwtConfig {
  secret: string;
  expiresIn: string;
}

interface StorageConfig {
  endpoint?: string;
  accessKey?: string;
  secretKey?: string;
  bucket: string;
  region?: string;
  useSSL: boolean;
}

interface RateLimitConfig {
  windowMs: number;
  defaultLimit: number;
}

export interface AppConfig {
  env: string;
  port: number;
  databaseUrl: string;
  postgres: PostgresConfig;
  redis: RedisConfig;
  services: ServiceConfig;
  jwt: JwtConfig;
  storage: StorageConfig;
  rateLimit: RateLimitConfig;
}

const toNumber = (value: string | undefined, fallback: number): number => {
  const parsed = Number(value);
  if (Number.isNaN(parsed)) {
    return fallback;
  }
  return parsed;
};

const buildDatabaseUrl = (): string => {
  if (process.env.DATABASE_URL) {
    return process.env.DATABASE_URL;
  }
  const host = process.env.POSTGRES_HOST || 'localhost';
  const port = toNumber(process.env.POSTGRES_PORT, 5432);
  const user = process.env.POSTGRES_USER || 'postgres';
  const password = process.env.POSTGRES_PASSWORD || 'postgres';
  const database = process.env.POSTGRES_DB || 'tts_stt';
  return `postgresql://${user}:${password}@${host}:${port}/${database}`;
};

const config: AppConfig = {
  env: process.env.NODE_ENV || 'development',
  port: toNumber(process.env.BACKEND_PORT, 4000),
  databaseUrl: buildDatabaseUrl(),
  postgres: {
    host: process.env.POSTGRES_HOST || 'localhost',
    port: toNumber(process.env.POSTGRES_PORT, 5432),
    user: process.env.POSTGRES_USER || 'postgres',
    password: process.env.POSTGRES_PASSWORD || 'postgres',
    database: process.env.POSTGRES_DB || 'tts_stt',
  },
  redis: {
    host: process.env.REDIS_HOST || 'localhost',
    port: toNumber(process.env.REDIS_PORT, 6379),
    password: process.env.REDIS_PASSWORD,
  },
  services: {
    ttsServiceUrl: process.env.TTS_SERVICE_URL || 'http://localhost:8001',
    sttServiceUrl: process.env.STT_SERVICE_URL || 'http://localhost:8002',
  },
  jwt: {
    secret: process.env.JWT_SECRET || 'dev-secret-change-me',
    expiresIn: process.env.JWT_EXPIRES_IN || '1h',
  },
  storage: {
    endpoint: process.env.S3_ENDPOINT,
    accessKey: process.env.S3_ACCESS_KEY,
    secretKey: process.env.S3_SECRET_KEY,
    bucket: process.env.S3_BUCKET || 'tts-stt-audio',
    region: process.env.S3_REGION || 'us-east-1',
    useSSL: process.env.S3_USE_SSL !== 'false',
  },
  rateLimit: {
    windowMs: toNumber(process.env.RATE_LIMIT_WINDOW_MS, 60_000),
    defaultLimit: toNumber(process.env.RATE_LIMIT_DEFAULT_LIMIT, 60),
  },
};

export default config;
