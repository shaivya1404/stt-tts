import { Request } from 'express';

import prisma from '../lib/prisma';

export interface OrgContext {
  orgId: string;
  userId?: string;
  apiKeyId?: string;
}

let cachedFallbackOrgId: string | null = null;
let fallbackOrgPromise: Promise<string> | null = null;

const ensureFallbackOrgId = async (): Promise<string> => {
  if (cachedFallbackOrgId) {
    return cachedFallbackOrgId;
  }

  if (fallbackOrgPromise) {
    return fallbackOrgPromise;
  }

  fallbackOrgPromise = (async () => {
    const explicitOrgId = process.env.DEV_ORG_ID?.trim();
    if (explicitOrgId) {
      const slug = process.env.DEV_ORG_SLUG?.trim() || `dev-${explicitOrgId.toLowerCase()}`;
      const name = process.env.DEV_ORG_NAME || 'Development Playground';
      const organization = await prisma.organization.upsert({
        where: { id: explicitOrgId },
        update: {},
        create: {
          id: explicitOrgId,
          name,
          slug,
        },
        select: { id: true },
      });
      return organization.id;
    }

    const existingOrg = await prisma.organization.findFirst({
      select: { id: true },
      orderBy: { createdAt: 'asc' },
    });

    if (existingOrg) {
      return existingOrg.id;
    }

    const slug = process.env.DEV_ORG_SLUG?.trim() || 'dev-playground';
    const name = process.env.DEV_ORG_NAME || 'Development Playground';

    const createdOrg = await prisma.organization.create({
      data: {
        name,
        slug,
      },
      select: { id: true },
    });

    return createdOrg.id;
  })();

  try {
    cachedFallbackOrgId = await fallbackOrgPromise;
    return cachedFallbackOrgId;
  } finally {
    fallbackOrgPromise = null;
  }
};

export const getOrgContext = async (req: Request): Promise<OrgContext> => {
  const orgId = req.auth?.orgId ?? (await ensureFallbackOrgId());

  return {
    orgId,
    userId: req.auth?.user?.id,
    apiKeyId: req.auth?.apiKey?.id,
  };
};
