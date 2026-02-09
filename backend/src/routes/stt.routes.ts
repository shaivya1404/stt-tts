import { Router } from 'express';
import multer from 'multer';

import { transcribe, transcribeRealtimeStub, batchTranscribe } from '../controllers/stt.controller';
import { optionalJwt } from '../middlewares/auth.middleware';
import { optionalApiKey, ensureScopeWhenPresent } from '../middlewares/apiKey.middleware';
import { rateLimit } from '../middlewares/rateLimit.middleware';

const upload = multer({ storage: multer.memoryStorage(), limits: { fileSize: 30 * 1024 * 1024 } });

const router = Router();

router.post(
  '/transcribe',
  optionalJwt,
  optionalApiKey,
  ensureScopeWhenPresent('stt'),
  rateLimit(),
  upload.single('audio_file'),
  transcribe,
);

router.post(
  '/transcribe-realtime',
  optionalJwt,
  optionalApiKey,
  ensureScopeWhenPresent('stt'),
  transcribeRealtimeStub,
);

router.post(
  '/batch-transcribe',
  optionalJwt,
  optionalApiKey,
  ensureScopeWhenPresent('stt'),
  rateLimit(),
  upload.array('audio_files'),
  batchTranscribe,
);

export default router;
