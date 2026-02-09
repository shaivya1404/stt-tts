import { Request, Response, NextFunction } from 'express';

import { SttService } from '../services/stt.service';
import { getOrgContext } from '../utils/requestContext';

type TranscribeResult = Awaited<ReturnType<typeof SttService.transcribe>>;

const parseLanguageHint = (value: unknown): string | undefined => {
  if (Array.isArray(value)) {
    return parseLanguageHint(value[0]);
  }

  if (typeof value === 'string') {
    const trimmed = value.trim();
    return trimmed.length > 0 ? trimmed : undefined;
  }

  return undefined;
};

const toResponsePayload = (result: TranscribeResult) => ({
  job_id: result.job.id,
  text: result.transcription.text,
  language: result.transcription.language,
  confidence: result.transcription.confidence,
  timestamps: result.transcription.timestamps,
});

export const transcribe = async (req: Request, res: Response, next: NextFunction): Promise<void> => {
  try {
    if (!req.file) {
      res.status(400).json({ message: 'audio_file is required' });
      return;
    }

    const context = await getOrgContext(req);
    const languageHint = parseLanguageHint(req.query.language_hint);

    const result = await SttService.transcribe({
      orgId: context.orgId,
      userId: context.userId,
      apiKeyId: context.apiKeyId,
      languageHint,
      file: req.file,
    });

    res.json(toResponsePayload(result));
  } catch (error) {
    next(error);
  }
};

export const transcribeRealtimeStub = (_req: Request, res: Response): void => {
  res.status(501).json({ message: 'Realtime transcription will be available in Phase 5' });
};

export const batchTranscribe = async (req: Request, res: Response, next: NextFunction): Promise<void> => {
  try {
    const files = Array.isArray(req.files) ? (req.files as Express.Multer.File[]) : [];
    if (!files.length) {
      res.status(400).json({ message: 'At least one audio_file is required' });
      return;
    }

    const context = await getOrgContext(req);
    const languageHint = parseLanguageHint(req.query.language_hint);

    const results = await SttService.batchTranscribe(
      {
        orgId: context.orgId,
        userId: context.userId,
        apiKeyId: context.apiKeyId,
      },
      files,
      languageHint,
    );

    res.json({ items: results.map(toResponsePayload) });
  } catch (error) {
    next(error);
  }
};
