import { Router } from 'express';

import authRoutes from './auth.routes';
import ttsRoutes from './tts.routes';
import sttRoutes from './stt.routes';
import modelsRoutes from './models.routes';
import analyticsRoutes from './analytics.routes';

const router = Router();

router.get('/', (_req, res) => {
  res.json({ message: 'TTS/STT API v1', status: 'running' });
});

router.use('/auth', authRoutes);
router.use('/tts', ttsRoutes);
router.use('/stt', sttRoutes);
router.use('/models', modelsRoutes);
router.use('/analytics', analyticsRoutes);

export default router;
