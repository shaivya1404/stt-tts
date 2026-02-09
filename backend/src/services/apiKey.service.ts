import crypto from 'crypto';

import prisma from '../lib/prisma';

const hashKey = (key: string): string => crypto.createHash('sha256').update(key).digest('hex');

const generateKey = (): string => `tts_${crypto.randomBytes(24).toString('hex')}`;

export const ApiKeyService = {
  async list(orgId: string) {
    return prisma.apiKey.findMany({
      where: { orgId },
      orderBy: { createdAt: 'desc' },
    });
  },

  async create(orgId: string, data: { name: string; scopes: string[]; rateLimitPerMinute?: number }) {
    const value = generateKey();
    const keyHash = hashKey(value);

    const apiKey = await prisma.apiKey.create({
      data: {
        orgId,
        keyHash,
        name: data.name,
        scopes: data.scopes,
        rateLimitPerMinute: data.rateLimitPerMinute,
      },
    });

    return { apiKey, value };
  },

  async revoke(orgId: string, apiKeyId: string) {
    const existing = await prisma.apiKey.findFirst({ where: { id: apiKeyId, orgId } });
    if (!existing) {
      const error = new Error('API key not found');
      (error as any).status = 404;
      throw error;
    }

    if (existing.revokedAt) {
      return existing;
    }

    return prisma.apiKey.update({
      where: { id: apiKeyId },
      data: { revokedAt: new Date() },
    });
  },
};
