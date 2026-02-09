import { NextFunction, Request, Response } from 'express';

import config from '../config';

interface RateBucket {
  count: number;
  expiresAt: number;
}

const buckets = new Map<string, RateBucket>();

const getKey = (req: Request): string => {
  if (req.auth?.apiKey) {
    return `api-key:${req.auth.apiKey.id}`;
  }
  if (req.auth?.orgId) {
    return `org:${req.auth.orgId}`;
  }
  return `ip:${req.ip}`;
};

export const rateLimit = (overrideLimit?: number) => (req: Request, res: Response, next: NextFunction): void => {
  const limit = overrideLimit || req.auth?.apiKey?.rateLimitPerMinute || config.rateLimit.defaultLimit;
  const windowMs = config.rateLimit.windowMs;
  const key = getKey(req);
  const now = Date.now();
  const bucket = buckets.get(key);

  if (!bucket || bucket.expiresAt <= now) {
    buckets.set(key, { count: 1, expiresAt: now + windowMs });
    next();
    return;
  }

  if (bucket.count >= limit) {
    const retryAfter = Math.ceil((bucket.expiresAt - now) / 1000);
    res.setHeader('Retry-After', retryAfter.toString());
    res.status(429).json({ message: 'Rate limit exceeded', retryAfter });
    return;
  }

  bucket.count += 1;
  next();
};
