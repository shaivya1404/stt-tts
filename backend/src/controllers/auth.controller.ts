import { Request, Response, NextFunction } from 'express';
import { z } from 'zod';

import { AuthService } from '../services/auth.service';
import { ApiKeyService } from '../services/apiKey.service';

const loginSchema = z.object({
  email: z.string().email(),
  password: z.string().min(6),
  orgSlug: z.string().optional(),
  org_slug: z.string().optional(),
});

const createApiKeySchema = z.object({
  name: z.string().min(1),
  scopes: z.array(z.enum(['tts', 'stt'])).min(1),
  rateLimitPerMinute: z.number().int().positive().optional(),
});

export const login = async (req: Request, res: Response, next: NextFunction): Promise<void> => {
  try {
    const payload = loginSchema.parse(req.body);
    const result = await AuthService.login({
      email: payload.email,
      password: payload.password,
      orgSlug: payload.orgSlug || payload.org_slug,
    });
    res.json(result);
  } catch (error) {
    if (error instanceof z.ZodError) {
      res.status(400).json({ message: error.message });
      return;
    }
    next(error);
  }
};

export const listApiKeys = async (req: Request, res: Response, next: NextFunction): Promise<void> => {
  try {
    const orgId = req.auth?.user?.orgId;
    if (!orgId) {
      res.status(401).json({ message: 'Authentication required' });
      return;
    }
    const apiKeys = await ApiKeyService.list(orgId);
    res.json(
      apiKeys.map((key) => ({
        id: key.id,
        name: key.name,
        scopes: key.scopes,
        rateLimitPerMinute: key.rateLimitPerMinute,
        createdAt: key.createdAt,
        lastUsedAt: key.lastUsedAt,
        revokedAt: key.revokedAt,
      })),
    );
  } catch (error) {
    next(error);
  }
};

export const createApiKey = async (req: Request, res: Response, next: NextFunction): Promise<void> => {
  try {
    const orgId = req.auth?.user?.orgId;
    if (!orgId) {
      res.status(401).json({ message: 'Authentication required' });
      return;
    }

    const payload = createApiKeySchema.parse(req.body);
    const { apiKey, value } = await ApiKeyService.create(orgId, payload);

    res.status(201).json({
      id: apiKey.id,
      name: apiKey.name,
      scopes: apiKey.scopes,
      rateLimitPerMinute: apiKey.rateLimitPerMinute,
      createdAt: apiKey.createdAt,
      apiKey: value,
    });
  } catch (error) {
    if (error instanceof z.ZodError) {
      res.status(400).json({ message: error.message });
      return;
    }
    next(error);
  }
};

export const revokeApiKey = async (req: Request, res: Response, next: NextFunction): Promise<void> => {
  try {
    const orgId = req.auth?.user?.orgId;
    if (!orgId) {
      res.status(401).json({ message: 'Authentication required' });
      return;
    }

    const { id } = req.params;
    const apiKey = await ApiKeyService.revoke(orgId, id);
    res.json({
      id: apiKey.id,
      revokedAt: apiKey.revokedAt,
    });
  } catch (error) {
    next(error);
  }
};
