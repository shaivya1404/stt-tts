import { MlModelStatus } from '@prisma/client';

import prisma from '../lib/prisma';
import { mlTtsClient } from './mlTtsClient.service';
import { mlSttClient } from './mlSttClient.service';

export const ModelsService = {
  async getStatus() {
    const models = await prisma.mlModel.findMany({ orderBy: { createdAt: 'asc' } });

    const [ttsHealth, sttHealth] = await Promise.allSettled([mlTtsClient.health(), mlSttClient.health()]);

    return {
      models,
      services: {
        tts: ttsHealth.status === 'fulfilled' ? ttsHealth.value : { status: 'unreachable' },
        stt: sttHealth.status === 'fulfilled' ? sttHealth.value : { status: 'unreachable' },
      },
    };
  },

  async reload() {
    await prisma.mlModel.updateMany({ data: { status: MlModelStatus.loading } });

    const [ttsInit, sttInit] = await Promise.allSettled([mlTtsClient.initialize(), mlSttClient.initialize()]);

    await prisma.mlModel.updateMany({
      where: { type: 'tts' },
      data: { status: ttsInit.status === 'fulfilled' ? MlModelStatus.active : MlModelStatus.error },
    });
    await prisma.mlModel.updateMany({
      where: { type: 'stt' },
      data: { status: sttInit.status === 'fulfilled' ? MlModelStatus.active : MlModelStatus.error },
    });

    return {
      tts: ttsInit.status === 'fulfilled' ? ttsInit.value : { status: 'error', detail: (ttsInit.reason as Error).message },
      stt: sttInit.status === 'fulfilled' ? sttInit.value : { status: 'error', detail: (sttInit.reason as Error).message },
    };
  },
};
