import { AudioFileType, JobStatus, UsageType } from '@prisma/client';

import prisma from '../lib/prisma';
import storageService from '../utils/storage';
import { OrgContext } from '../utils/requestContext';
import { mlSttClient } from './mlSttClient.service';
import type { MlSttTranscribeResult } from './mlSttClient.service';
import { UsageService } from './usage.service';

interface TranscribeInput {
  orgId: string;
  userId?: string;
  apiKeyId?: string;
  languageHint?: string;
  file: Express.Multer.File;
}

const extractExtension = (fileName: string | undefined): string | undefined => {
  if (!fileName) {
    return undefined;
  }
  const parts = fileName.split('.');
  if (parts.length < 2) {
    return undefined;
  }
  return parts.pop();
};

const deriveDurationSeconds = (mlResponse: MlSttTranscribeResult): number | undefined => {
  const metaDuration = Number(mlResponse.meta?.duration_seconds);
  if (!Number.isNaN(metaDuration) && metaDuration > 0) {
    return metaDuration;
  }
  const lastTimestamp = mlResponse.timestamps?.[mlResponse.timestamps.length - 1];
  return lastTimestamp?.end;
};

export const SttService = {
  async transcribe(input: TranscribeInput) {
    const storagePath = await storageService.uploadBuffer(input.file.buffer, input.file.mimetype, {
      prefix: `stt-input/${input.orgId}`,
      extension: extractExtension(input.file.originalname),
    });

    const audioFile = await prisma.audioFile.create({
      data: {
        orgId: input.orgId,
        userId: input.userId,
        type: AudioFileType.stt_input,
        storagePath,
        mimeType: input.file.mimetype,
        sizeBytes: input.file.size,
      },
    });

    const job = await prisma.sttJob.create({
      data: {
        orgId: input.orgId,
        userId: input.userId,
        apiKeyId: input.apiKeyId,
        inputAudioFileId: audioFile.id,
        status: JobStatus.processing,
        modelUsed: 'conformer_rnnt_indic',
      },
    });

    try {
      const mlResponse = await mlSttClient.transcribeWithMl({
        buffer: input.file.buffer,
        fileName: input.file.originalname || `${job.id}.wav`,
        mimeType: input.file.mimetype,
        languageHint: input.languageHint,
      });

      const transcription = await prisma.transcription.create({
        data: {
          sttJobId: job.id,
          text: mlResponse.text,
          language: mlResponse.language,
          confidence: mlResponse.confidence,
          timestamps: mlResponse.timestamps,
        },
      });

      const durationSeconds = deriveDurationSeconds(mlResponse);
      if (durationSeconds) {
        await prisma.audioFile.update({
          where: { id: audioFile.id },
          data: { durationSeconds },
        });
      }

      const updatedJob = await prisma.sttJob.update({
        where: { id: job.id },
        data: {
          status: JobStatus.completed,
          languageDetected: mlResponse.language,
          modelUsed: mlResponse.modelUsed || job.modelUsed,
          completedAt: new Date(),
        },
      });

      await UsageService.record({
        orgId: input.orgId,
        apiKeyId: input.apiKeyId,
        type: UsageType.stt,
        units: durationSeconds ?? 0,
        metadata: {
          jobId: job.id,
          inputAudioFileId: audioFile.id,
          language: mlResponse.language,
          modelUsed: mlResponse.modelUsed || job.modelUsed,
          durationSeconds: durationSeconds ?? 0,
        },
      });

      return { job: updatedJob, transcription };
    } catch (error) {
      await prisma.sttJob.update({
        where: { id: job.id },
        data: {
          status: JobStatus.failed,
          errorMessage: error instanceof Error ? error.message : 'STT transcription failed',
        },
      });
      throw error;
    }
  },

  async batchTranscribe(context: OrgContext, files: Express.Multer.File[], languageHint?: string) {
    const results: Array<Awaited<ReturnType<typeof SttService.transcribe>>> = [];
    // eslint-disable-next-line no-restricted-syntax
    for (const file of files) {
      // eslint-disable-next-line no-await-in-loop
      const result = await SttService.transcribe({
        orgId: context.orgId,
        userId: context.userId,
        apiKeyId: context.apiKeyId,
        languageHint,
        file,
      });
      results.push(result);
    }
    return results;
  },
};
