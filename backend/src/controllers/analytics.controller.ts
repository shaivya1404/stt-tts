import { Request, Response, NextFunction } from 'express';

import { AnalyticsService } from '../services/analytics.service';
import { getOrgContext } from '../utils/requestContext';

export const getUsage = async (req: Request, res: Response, next: NextFunction): Promise<void> => {
  try {
    const context = await getOrgContext(req);
    const usage = await AnalyticsService.usage(context.orgId);
    res.json(usage);
  } catch (error) {
    next(error);
  }
};
