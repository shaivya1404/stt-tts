import axios from 'axios';

import config from '../config';

export interface TtsSynthesizeInput {
  text: string;
  language: string;
  voiceId?: string;
  emotion?: string;
  speed?: number;
}

export interface TtsSynthesizeResult {
  audioPath: string | null;
  duration: number | null;
  status: string;
  meta?: Record<string, any>;
}

const BASE_URL = config.services.ttsServiceUrl;

export const mlTtsClient = {
  async synthesize(payload: TtsSynthesizeInput): Promise<TtsSynthesizeResult> {
    const response = await axios.post(`${BASE_URL}/ml/tts/predict`, {
      text: payload.text,
      language: payload.language,
      voice_id: payload.voiceId,
      emotion: payload.emotion,
      speed: payload.speed,
    });

    const data = response.data;
    return {
      audioPath: data.audio_path ?? null,
      duration: typeof data.duration === 'number' ? data.duration : null,
      status: data.status ?? 'success',
      meta: data.meta ?? {},
    };
  },

  async health(): Promise<{ status: string }> {
    const response = await axios.get(`${BASE_URL}/ml/tts/health`);
    return response.data;
  },

  async initialize(): Promise<{ status: string; detail?: string }> {
    const response = await axios.post(`${BASE_URL}/ml/tts/initialize`);
    return response.data;
  },
};
