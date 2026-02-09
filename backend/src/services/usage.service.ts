import { Prisma, UsageType } from '@prisma/client';

import prisma from '../lib/prisma';

export interface RecordUsageInput {
  orgId: string;
  apiKeyId?: string;
  type: UsageType;
  units: number;
  metadata?: Prisma.InputJsonValue;
}

export const UsageService = {
  async record(input: RecordUsageInput) {
    return prisma.usageRecord.create({
      data: {
        orgId: input.orgId,
        apiKeyId: input.apiKeyId,
        type: input.type,
        units: input.units,
        metadata: input.metadata,
      },
    });
  },
};
