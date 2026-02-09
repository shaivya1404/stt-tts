import { ApiKey, Organization, User } from '@prisma/client';

declare global {
  namespace Express {
    interface AuthContext {
      user?: User;
      organization?: Organization;
      apiKey?: ApiKey;
      orgId?: string;
    }

    interface Request {
      auth?: AuthContext;
    }
  }
}

export {};
