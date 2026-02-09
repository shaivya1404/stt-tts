import { Router } from 'express';

import { getUsage } from '../controllers/analytics.controller';

const router = Router();

router.get('/usage', getUsage);

export default router;
