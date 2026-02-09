import bcrypt from 'bcryptjs';
import jwt, { SignOptions } from 'jsonwebtoken';
import { Prisma } from '@prisma/client';

import config from '../config';
import prisma from '../lib/prisma';

export interface LoginPayload {
  email: string;
  password: string;
  orgSlug?: string;
}

export interface LoginResult {
  token: string;
  user: {
    id: string;
    email: string;
    orgId: string;
    role: string;
  };
}

const buildToken = (user: { id: string; orgId: string; role: string; email: string }): string => {
  const expiresIn = config.jwt.expiresIn as SignOptions['expiresIn'];
  return jwt.sign(
    {
      sub: user.id,
      orgId: user.orgId,
      role: user.role,
      email: user.email,
    },
    config.jwt.secret,
    { expiresIn },
  );
};

export const AuthService = {
  async login(payload: LoginPayload): Promise<LoginResult> {
    const where: Prisma.UserWhereInput = {
      email: payload.email,
    };

    if (payload.orgSlug) {
      where.organization = { slug: payload.orgSlug };
    }

    const user = await prisma.user.findFirst({ where });

    if (!user) {
      const error = new Error('Invalid credentials');
      (error as any).status = 401;
      throw error;
    }

    const isValid = await bcrypt.compare(payload.password, user.passwordHash);
    if (!isValid) {
      const error = new Error('Invalid credentials');
      (error as any).status = 401;
      throw error;
    }

    return {
      token: buildToken(user),
      user: {
        id: user.id,
        email: user.email,
        orgId: user.orgId,
        role: user.role,
      },
    };
  },
};
