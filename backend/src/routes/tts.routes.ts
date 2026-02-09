import { Router } from 'express';
import multer from 'multer';

import { synthesize, synthesizeBatch, listVoices, voiceClone } from '../controllers/tts.controller';
import { optionalJwt } from '../middlewares/auth.middleware';
import { optionalApiKey, ensureScopeWhenPresent } from '../middlewares/apiKey.middleware';
import { rateLimit } from '../middlewares/rateLimit.middleware';

const upload = multer({ storage: multer.memoryStorage(), limits: { fileSize: 25 * 1024 * 1024 } });

const router = Router();

router.post(
  '/synthesize',
  optionalJwt,
  optionalApiKey,
  ensureScopeWhenPresent('tts'),
  rateLimit(),
  synthesize,
);
router.post(
  '/synthesize-batch',
  optionalJwt,
  optionalApiKey,
  ensureScopeWhenPresent('tts'),
  rateLimit(),
  synthesizeBatch,
);
router.get(
  '/voices',
  optionalJwt,
  optionalApiKey,
  ensureScopeWhenPresent('tts'),
  listVoices,
);
router.post(
  '/voice-clone',
  optionalJwt,
  optionalApiKey,
  ensureScopeWhenPresent('tts'),
  upload.single('audio_sample'),
  voiceClone,
);

export default router;
