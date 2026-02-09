import { Request, Response, NextFunction } from 'express';

interface HttpError extends Error {
  status?: number;
}

const errorHandler = (
  err: HttpError,
  _req: Request,
  res: Response,
  _next: NextFunction,
): void => {
  const statusCode = err.status || 500;
  const message = err.message || 'Internal Server Error';

  // eslint-disable-next-line no-console
  console.error(err);

  res.status(statusCode).json({ message, status: 'error' });
};

export default errorHandler;
