import { Router } from 'express';
import { UserRole } from '@prisma/client';

import { login, listApiKeys, createApiKey, revokeApiKey } from '../controllers/auth.controller';
import { requireJwt, requireRole } from '../middlewares/auth.middleware';

const router = Router();

router.post('/login', login);

router.get('/api-keys', requireJwt, requireRole([UserRole.owner, UserRole.admin]), listApiKeys);
router.post('/api-keys', requireJwt, requireRole([UserRole.owner, UserRole.admin]), createApiKey);
router.post('/api-keys/:id/revoke', requireJwt, requireRole([UserRole.owner, UserRole.admin]), revokeApiKey);

export default router;
