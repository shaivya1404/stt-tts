import { AudioFileType, JobStatus, VoiceProfileStatus } from '@prisma/client';

import prisma from '../lib/prisma';
import { mlTtsClient } from './mlTtsClient.service';
import storageService from '../utils/storage';

interface SynthesizeInput {
  orgId: string;
  userId?: string;
  apiKeyId?: string;
  text: string;
  language: string;
  voiceId?: string;
  emotion?: string;
  speed?: number;
}

interface OrgContext {
  orgId: string;
  userId?: string;
  apiKeyId?: string;
}

export interface TtsSynthesizeResult {
  jobId: string;
  status: JobStatus;
  audioUrl: string | null;
  duration: number | null;
}

const resolveCharacterUnits = (text: string, meta?: Record<string, any>): number => {
  const rawValue = meta?.char_count ?? meta?.characters ?? meta?.charCount;
  if (typeof rawValue === 'number' && Number.isFinite(rawValue)) {
    return rawValue;
  }
  if (typeof rawValue === 'string') {
    const parsed = Number(rawValue);
    if (!Number.isNaN(parsed)) {
      return parsed;
    }
  }
  return text.length;
};

export const TtsService = {
  async synthesize(input: SynthesizeInput): Promise<TtsSynthesizeResult> {
    const job = await prisma.ttsJob.create({
      data: {
        orgId: input.orgId,
        userId: input.userId,
        apiKeyId: input.apiKeyId,
        inputText: input.text,
        language: input.language,
        voiceProfileId: input.voiceId,
        emotion: input.emotion,
        speed: input.speed,
        status: JobStatus.processing,
      },
    });

    try {
      const mlResponse = await mlTtsClient.synthesize({
        text: input.text,
        language: input.language,
        voiceId: input.voiceId,
        emotion: input.emotion,
        speed: input.speed,
      });

      const pipelineSucceeded = mlResponse.status?.toLowerCase() === 'success' && Boolean(mlResponse.audioPath);
      if (!pipelineSucceeded || !mlResponse.audioPath) {
        const errorMessage = (mlResponse.meta?.error as string) || 'TTS synthesis failed';
        throw new Error(errorMessage);
      }

      const audioFile = await prisma.audioFile.create({
        data: {
          orgId: input.orgId,
          userId: input.userId,
          type: AudioFileType.tts_output,
          storagePath: mlResponse.audioPath,
          mimeType: 'audio/wav',
          durationSeconds: mlResponse.duration ?? undefined,
        },
      });

      const completedJob = await prisma.ttsJob.update({
        where: { id: job.id },
        data: {
          status: JobStatus.completed,
          resultAudioFileId: audioFile.id,
          completedAt: new Date(),
        },
      });

      const units = resolveCharacterUnits(input.text, mlResponse.meta);
      await prisma.usageRecord.create({
        data: {
          orgId: input.orgId,
          apiKeyId: input.apiKeyId,
          type: 'tts',
          units,
          metadata: {
            jobId: job.id,
            voiceProfileId: input.voiceId,
            charCount: units,
          },
        },
      });

      return {
        jobId: completedJob.id,
        status: completedJob.status,
        audioUrl: audioFile.storagePath,
        duration: mlResponse.duration ?? null,
      };
    } catch (error) {
      await prisma.ttsJob.update({
        where: { id: job.id },
        data: {
          status: JobStatus.failed,
          errorMessage: error instanceof Error ? error.message : 'TTS synthesis failed',
        },
      });
      throw error;
    }
  },

  async synthesizeBatch(
    orgContext: OrgContext,
    requests: Array<Omit<SynthesizeInput, 'orgId' | 'userId' | 'apiKeyId'>>,
  ): Promise<TtsSynthesizeResult[]> {
    const results: TtsSynthesizeResult[] = [];
    // eslint-disable-next-line no-restricted-syntax
    for (const item of requests) {
      // eslint-disable-next-line no-await-in-loop
      const result = await TtsService.synthesize({
        ...item,
        orgId: orgContext.orgId,
        userId: orgContext.userId,
        apiKeyId: orgContext.apiKeyId,
      });
      results.push(result);
    }
    return results;
  },

  async listVoices(orgId: string) {
    return prisma.voiceProfile.findMany({
      where: { orgId },
      orderBy: { createdAt: 'desc' },
      select: {
        id: true,
        name: true,
        language: true,
        gender: true,
        description: true,
        status: true,
      },
    });
  },

  async cloneVoice(input: {
    orgId: string;
    userId?: string;
    file: { buffer: Buffer; mimetype: string; size: number; originalname: string };
    name: string;
    language: string;
    gender?: string;
    description?: string;
    baseModel?: string;
  }) {
    const extension = input.file.originalname.split('.').pop();
    const storageKey = await storageService.uploadBuffer(input.file.buffer, input.file.mimetype, {
      prefix: `voice-clone/${input.orgId}`,
      extension,
    });

    await prisma.audioFile.create({
      data: {
        orgId: input.orgId,
        userId: input.userId,
        type: AudioFileType.voice_clone_sample,
        storagePath: storageKey,
        mimeType: input.file.mimetype,
        sizeBytes: input.file.size,
      },
    });

    const placeholderEmbeddingPath = `voice-embeddings/${input.orgId}/${Date.now()}-pending.vec`;

    return prisma.voiceProfile.create({
      data: {
        orgId: input.orgId,
        name: input.name,
        language: input.language,
        gender: input.gender,
        description: input.description,
        baseModel: input.baseModel || 'vits_multispkr_indic_v1',
        speakerEmbeddingPath: placeholderEmbeddingPath,
        status: VoiceProfileStatus.training,
      },
    });
  },
};
