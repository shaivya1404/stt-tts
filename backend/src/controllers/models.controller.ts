import { Request, Response, NextFunction } from 'express';

import { ModelsService } from '../services/models.service';

export const getModelsStatus = async (_req: Request, res: Response, next: NextFunction): Promise<void> => {
  try {
    const data = await ModelsService.getStatus();
    res.json(data);
  } catch (error) {
    next(error);
  }
};

export const reloadModels = async (_req: Request, res: Response, next: NextFunction): Promise<void> => {
  try {
    const result = await ModelsService.reload();
    res.json(result);
  } catch (error) {
    next(error);
  }
};
