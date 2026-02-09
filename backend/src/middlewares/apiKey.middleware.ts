import crypto from 'crypto';
import { NextFunction, Request, Response } from 'express';

import prisma from '../lib/prisma';

const hashKey = (key: string): string => crypto.createHash('sha256').update(key).digest('hex');

const ensureAuthContainer = (req: Request): NonNullable<Request['auth']> => {
  if (!req.auth) {
    req.auth = {};
  }
  return req.auth;
};

const attachApiKey = async (req: Request): Promise<void> => {
  const apiKeyHeader = req.header('x-api-key');
  if (!apiKeyHeader) {
    return;
  }

  const hashed = hashKey(apiKeyHeader.trim());
  const apiKeyRecord = await prisma.apiKey.findUnique({
    where: { keyHash: hashed },
    include: { organization: true },
  });

  if (!apiKeyRecord || apiKeyRecord.revokedAt) {
    const error = new Error('Invalid API key');
    (error as any).status = 401;
    throw error;
  }

  const auth = ensureAuthContainer(req);
  auth.apiKey = apiKeyRecord;
  auth.organization = apiKeyRecord.organization;
  auth.orgId = apiKeyRecord.orgId;

  await prisma.apiKey.update({
    where: { id: apiKeyRecord.id },
    data: { lastUsedAt: new Date() },
  });
};

export const optionalApiKey = async (req: Request, _res: Response, next: NextFunction): Promise<void> => {
  try {
    await attachApiKey(req);
    next();
  } catch (error) {
    next(error);
  }
};

export const requireApiKey = async (req: Request, res: Response, next: NextFunction): Promise<void> => {
  try {
    await attachApiKey(req);
    if (!req.auth?.apiKey) {
      res.status(401).json({ message: 'API key required' });
      return;
    }
    next();
  } catch (error) {
    next(error);
  }
};

export const requireOrgContext = (req: Request, res: Response, next: NextFunction): void => {
  if (req.auth?.orgId) {
    next();
    return;
  }
  res.status(401).json({ message: 'Organization context missing' });
};

export const requireApiKeyScopes = (scopes: string[]) => (
  req: Request,
  res: Response,
  next: NextFunction,
): void => {
  if (!req.auth?.apiKey) {
    res.status(401).json({ message: 'API key required' });
    return;
  }

  if (scopes.length === 0) {
    next();
    return;
  }

  const hasScope = scopes.some((scope) => req.auth?.apiKey?.scopes.includes(scope));
  if (!hasScope) {
    res.status(403).json({ message: 'API key does not have required scope' });
    return;
  }
  next();
};

export const ensureScopeWhenPresent = (scope: string) => (
  req: Request,
  res: Response,
  next: NextFunction,
): void => {
  if (!req.auth?.apiKey) {
    next();
    return;
  }

  if (!req.auth.apiKey.scopes.includes(scope)) {
    res.status(403).json({ message: `API key missing required scope: ${scope}` });
    return;
  }

  next();
};
