import axios from 'axios';
import FormData from 'form-data';

import config from '../config';

export interface MlSttTranscribeInput {
  buffer: Buffer;
  fileName: string;
  mimeType: string;
  languageHint?: string;
}

export interface MlSttTranscribeResult {
  text: string;
  language: string;
  confidence: number;
  timestamps: Array<{ start: number; end: number; word?: string }>;
  meta?: Record<string, any>;
  modelUsed?: string;
}

const BASE_URL = config.services.sttServiceUrl;

export const mlSttClient = {
  async transcribeWithMl(input: MlSttTranscribeInput): Promise<MlSttTranscribeResult> {
    const formData = new FormData();
    formData.append('file', input.buffer, {
      filename: input.fileName,
      contentType: input.mimeType,
    });

    if (input.languageHint) {
      formData.append('language_hint', input.languageHint);
    }

    const response = await axios.post(`${BASE_URL}/ml/stt/transcribe`, formData, {
      headers: formData.getHeaders(),
      maxBodyLength: Infinity,
    });

    const { data } = response;
    return {
      text: data.text,
      language: data.language,
      confidence: data.confidence,
      timestamps: data.timestamps || [],
      meta: data.meta,
      modelUsed: data.modelUsed,
    };
  },

  async health(): Promise<{ status: string }> {
    const response = await axios.get(`${BASE_URL}/ml/stt/health`);
    return response.data;
  },

  async initialize(): Promise<{ status: string; detail?: string }> {
    const response = await axios.post(`${BASE_URL}/ml/stt/initialize`);
    return response.data;
  },
};
