import { Request, Response, NextFunction } from 'express';
import { z } from 'zod';

import { TtsService } from '../services/tts.service';
import { getOrgContext } from '../utils/requestContext';

const synthSchema = z.object({
  text: z.string().min(1),
  language: z.string().min(2),
  voice_id: z.string().optional(),
  voiceId: z.string().optional(),
  emotion: z.string().optional(),
  speed: z.number().positive().max(5).optional(),
});

const batchSchema = z.object({
  items: z.array(synthSchema).min(1),
});

const voiceCloneSchema = z.object({
  name: z.string().min(2),
  language: z.string().min(2),
  gender: z.string().optional(),
  description: z.string().optional(),
  baseModel: z.string().optional(),
});

export const synthesize = async (req: Request, res: Response, next: NextFunction): Promise<void> => {
  try {
    const payload = synthSchema.parse(req.body);
    const context = await getOrgContext(req);
    const result = await TtsService.synthesize({
      ...context,
      text: payload.text,
      language: payload.language,
      voiceId: payload.voiceId || payload.voice_id,
      emotion: payload.emotion,
      speed: payload.speed,
    });

    res.json({
      job_id: result.jobId,
      audio_url: result.audioUrl,
      duration: result.duration,
      status: result.status,
    });
  } catch (error) {
    if (error instanceof z.ZodError) {
      res.status(400).json({ message: error.message });
      return;
    }
    next(error);
  }
};

export const synthesizeBatch = async (req: Request, res: Response, next: NextFunction): Promise<void> => {
  try {
    const payload = batchSchema.parse(req.body);
    const context = await getOrgContext(req);

    const normalizedItems = payload.items.map((item) => ({
      text: item.text,
      language: item.language,
      voiceId: item.voiceId || item.voice_id,
      emotion: item.emotion,
      speed: item.speed,
    }));

    const items = await TtsService.synthesizeBatch(context, normalizedItems);

    res.json({
      items: items.map((item) => ({
        job_id: item.jobId,
        audio_url: item.audioUrl,
        duration: item.duration,
        status: item.status,
      })),
    });
  } catch (error) {
    if (error instanceof z.ZodError) {
      res.status(400).json({ message: error.message });
      return;
    }
    next(error);
  }
};

export const listVoices = async (req: Request, res: Response, next: NextFunction): Promise<void> => {
  try {
    const context = await getOrgContext(req);
    const voices = await TtsService.listVoices(context.orgId);
    res.json(voices);
  } catch (error) {
    next(error);
  }
};

export const voiceClone = async (req: Request, res: Response, next: NextFunction): Promise<void> => {
  try {
    if (!req.file) {
      res.status(400).json({ message: 'audio_sample file is required' });
      return;
    }

    const payload = voiceCloneSchema.parse(req.body);
    const context = await getOrgContext(req);

    const voice = await TtsService.cloneVoice({
      orgId: context.orgId,
      userId: context.userId,
      file: {
        buffer: req.file.buffer,
        mimetype: req.file.mimetype,
        originalname: req.file.originalname,
        size: req.file.size,
      },
      name: payload.name,
      language: payload.language,
      gender: payload.gender,
      description: payload.description,
      baseModel: payload.baseModel,
    });

    res.status(201).json(voice);
  } catch (error) {
    if (error instanceof z.ZodError) {
      res.status(400).json({ message: error.message });
      return;
    }
    next(error);
  }
};
