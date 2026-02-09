import { Router } from 'express';

import { getModelsStatus, reloadModels } from '../controllers/models.controller';

const router = Router();

router.get('/status', getModelsStatus);
router.post('/reload', reloadModels);

export default router;
