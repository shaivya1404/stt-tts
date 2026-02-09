import { NextFunction, Request, Response } from 'express';
import jwt from 'jsonwebtoken';
import { UserRole } from '@prisma/client';

import config from '../config';
import prisma from '../lib/prisma';

interface JwtPayload {
  sub: string;
  orgId: string;
  role: UserRole;
  email: string;
}

const ensureAuthContainer = (req: Request): NonNullable<Request['auth']> => {
  if (!req.auth) {
    req.auth = {};
  }
  return req.auth;
};

const attachJwtToRequest = async (req: Request): Promise<void> => {
  const header = req.headers.authorization;
  if (!header) {
    return;
  }

  const [scheme, token] = header.split(' ');
  if (scheme !== 'Bearer' || !token) {
    const error = new Error('Invalid Authorization header format');
    (error as any).status = 401;
    throw error;
  }

  const payload = jwt.verify(token, config.jwt.secret) as JwtPayload;

  const user = await prisma.user.findUnique({
    where: { id: payload.sub },
    include: { organization: true },
  });

  if (!user || user.orgId !== payload.orgId) {
    const error = new Error('Invalid authentication token');
    (error as any).status = 401;
    throw error;
  }

  const auth = ensureAuthContainer(req);
  auth.user = user;
  auth.organization = user.organization;
  auth.orgId = user.orgId;
};

export const optionalJwt = async (req: Request, _res: Response, next: NextFunction): Promise<void> => {
  try {
    await attachJwtToRequest(req);
    next();
  } catch (error) {
    next(error);
  }
};

export const requireJwt = async (req: Request, res: Response, next: NextFunction): Promise<void> => {
  try {
    await attachJwtToRequest(req);
    if (!req.auth?.user) {
      res.status(401).json({ message: 'Authentication required' });
      return;
    }
    next();
  } catch (error) {
    next(error);
  }
};

export const requireRole = (roles: UserRole[]) => (
  req: Request,
  res: Response,
  next: NextFunction,
): void => {
  if (!req.auth?.user) {
    res.status(401).json({ message: 'Authentication required' });
    return;
  }

  if (!roles.includes(req.auth.user.role)) {
    res.status(403).json({ message: 'Forbidden' });
    return;
  }

  next();
};

export const requireAuthOrApiKey = (req: Request, res: Response, next: NextFunction): void => {
  if (req.auth?.user || req.auth?.apiKey) {
    next();
    return;
  }

  res.status(401).json({ message: 'Authentication or API key required' });
};
