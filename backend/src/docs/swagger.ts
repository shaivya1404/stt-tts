import swaggerJsdoc from 'swagger-jsdoc';

const swaggerDefinition = {
  openapi: '3.0.1',
  info: {
    title: 'Enterprise TTS/STT Platform API',
    version: '0.2.0',
    description:
      'REST API for the enterprise-grade TTS/STT platform. Endpoints shown below require either JWT (Bearer) authentication or API keys as noted.',
  },
  servers: [{ url: '/api/v1' }],
  components: {
    securitySchemes: {
      bearerAuth: {
        type: 'http',
        scheme: 'bearer',
        bearerFormat: 'JWT',
      },
      apiKey: {
        type: 'apiKey',
        in: 'header',
        name: 'x-api-key',
      },
    },
    schemas: {
      TtsSynthesizeRequest: {
        type: 'object',
        required: ['text', 'language'],
        properties: {
          text: { type: 'string' },
          language: { type: 'string', example: 'en-US' },
          voice_id: { type: 'string' },
          emotion: { type: 'string' },
          speed: { type: 'number', example: 1.0 },
        },
      },
      TtsSynthesizeResponse: {
        type: 'object',
        properties: {
          job_id: { type: 'string' },
          audio_url: { type: 'string' },
          duration: { type: 'number', nullable: true },
          status: { type: 'string', enum: ['queued', 'processing', 'completed', 'failed'] },
        },
      },
      SttTranscribeResponse: {
        type: 'object',
        properties: {
          job_id: { type: 'string' },
          text: { type: 'string' },
          language: { type: 'string' },
          confidence: { type: 'number' },
          timestamps: {
            type: 'array',
            items: {
              type: 'object',
              properties: {
                start: { type: 'number' },
                end: { type: 'number' },
                word: { type: 'string' },
              },
            },
          },
        },
      },
      UsageAnalyticsResponse: {
        type: 'object',
        properties: {
          totals: {
            type: 'object',
            properties: {
              totalTtsChars: { type: 'number' },
              totalSttSeconds: { type: 'number' },
            },
          },
          last7Days: {
            type: 'array',
            items: {
              type: 'object',
              properties: {
                date: { type: 'string', format: 'date' },
                tts: { type: 'number' },
                stt: { type: 'number' },
              },
            },
          },
          last30Days: {
            type: 'array',
            items: {
              type: 'object',
              properties: {
                date: { type: 'string', format: 'date' },
                tts: { type: 'number' },
                stt: { type: 'number' },
              },
            },
          },
        },
      },
    },
  },
  paths: {
    '/auth/login': {
      post: {
        summary: 'Login with email/password',
        requestBody: {
          required: true,
          content: {
            'application/json': {
              schema: {
                type: 'object',
                required: ['email', 'password'],
                properties: {
                  email: { type: 'string', format: 'email' },
                  password: { type: 'string' },
                  orgSlug: { type: 'string', description: 'Optional org slug if the email exists in multiple orgs' },
                },
              },
            },
          },
        },
        responses: {
          200: {
            description: 'JWT token response',
          },
        },
      },
    },
    '/tts/synthesize': {
      post: {
        summary: 'Synthesize a single text payload into audio',
        security: [{ bearerAuth: [] }, { apiKey: [] }],
        requestBody: {
          required: true,
          content: {
            'application/json': {
              schema: { $ref: '#/components/schemas/TtsSynthesizeRequest' },
            },
          },
        },
        responses: {
          200: {
            description: 'TTS synthesis result',
            content: {
              'application/json': {
                schema: { $ref: '#/components/schemas/TtsSynthesizeResponse' },
              },
            },
          },
        },
      },
    },
    '/tts/synthesize-batch': {
      post: {
        summary: 'Submit multiple synthesis requests in a single payload',
        security: [{ bearerAuth: [] }, { apiKey: [] }],
        requestBody: {
          required: true,
          content: {
            'application/json': {
              schema: {
                type: 'object',
                properties: {
                  items: {
                    type: 'array',
                    items: { $ref: '#/components/schemas/TtsSynthesizeRequest' },
                  },
                },
              },
            },
          },
        },
        responses: { 200: { description: 'Batch synthesis accepted' } },
      },
    },
    '/tts/voices': {
      get: {
        summary: 'List voice profiles for organization',
        security: [{ bearerAuth: [] }],
        responses: { 200: { description: 'Voice profile list' } },
      },
    },
    '/tts/voice-clone': {
      post: {
        summary: 'Upload a sample to kick off voice cloning',
        security: [{ bearerAuth: [] }],
        requestBody: {
          required: true,
          content: {
            'multipart/form-data': {
              schema: {
                type: 'object',
                properties: {
                  audio_sample: { type: 'string', format: 'binary' },
                  name: { type: 'string' },
                  language: { type: 'string' },
                },
              },
            },
          },
        },
        responses: { 201: { description: 'Voice profile created' } },
      },
    },
    '/stt/transcribe': {
      post: {
        summary: 'Upload audio and receive a transcription',
        security: [{ bearerAuth: [] }, { apiKey: [] }],
        requestBody: {
          required: true,
          content: {
            'multipart/form-data': {
              schema: {
                type: 'object',
                properties: {
                  audio_file: { type: 'string', format: 'binary' },
                  language: { type: 'string' },
                },
              },
            },
          },
        },
        responses: {
          200: {
            description: 'Transcription payload',
            content: {
              'application/json': {
                schema: { $ref: '#/components/schemas/SttTranscribeResponse' },
              },
            },
          },
        },
      },
    },
    '/stt/transcribe-realtime': {
      post: {
        summary: 'Realtime transcription placeholder',
        security: [{ bearerAuth: [] }, { apiKey: [] }],
        responses: { 501: { description: 'Not implemented yet' } },
      },
    },
    '/stt/batch-transcribe': {
      post: {
        summary: 'Submit multiple audio files for batch transcription',
        security: [{ bearerAuth: [] }, { apiKey: [] }],
        requestBody: {
          required: true,
          content: {
            'multipart/form-data': {
              schema: {
                type: 'object',
                properties: {
                  audio_files: {
                    type: 'array',
                    items: { type: 'string', format: 'binary' },
                  },
                },
              },
            },
          },
        },
        responses: { 200: { description: 'Batch transcription response' } },
      },
    },
    '/models/status': {
      get: {
        summary: 'Inspect registered ML models and service health',
        security: [{ bearerAuth: [] }],
        responses: { 200: { description: 'Model inventory with service status' } },
      },
    },
    '/models/reload': {
      post: {
        summary: 'Trigger ML model reload sequence',
        security: [{ bearerAuth: [] }],
        responses: { 200: { description: 'Reload summary' } },
      },
    },
    '/analytics/usage': {
      get: {
        summary: 'Usage analytics rollups for current organization',
        security: [{ bearerAuth: [] }],
        responses: {
          200: {
            description: 'Usage analytics response',
            content: {
              'application/json': {
                schema: { $ref: '#/components/schemas/UsageAnalyticsResponse' },
              },
            },
          },
        },
      },
    },
  },
};

const swaggerSpec = swaggerJsdoc({
  definition: swaggerDefinition,
  apis: [],
});

export default swaggerSpec;
