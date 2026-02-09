import express from 'express';
import swaggerUi from 'swagger-ui-express';

import routes from './routes';
import errorHandler from './middlewares/error.middleware';
import prisma from './lib/prisma';
import swaggerSpec from './docs/swagger';

const app = express();

app.use(express.json({ limit: '5mb' }));
app.use(express.urlencoded({ extended: true }));

app.get('/health', async (_req, res) => {
  try {
    await prisma.$queryRaw`SELECT 1`;
    res.json({ status: 'ok', service: 'backend-api', db: 'ok' });
  } catch (error) {
    res.status(500).json({ status: 'error', service: 'backend-api', db: 'error', message: (error as Error).message });
  }
});

app.use('/api/v1/docs', swaggerUi.serve, swaggerUi.setup(swaggerSpec));
app.use('/api/v1', routes);

app.use(errorHandler);

export default app;
