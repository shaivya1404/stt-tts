import { S3Client, PutObjectCommand, GetObjectCommand } from '@aws-sdk/client-s3';
import { getSignedUrl } from '@aws-sdk/s3-request-presigner';
import crypto from 'crypto';

import config from '../config';

class StorageService {
  private client?: S3Client;

  private enabled: boolean;

  constructor() {
    this.enabled = Boolean(config.storage.bucket && config.storage.accessKey && config.storage.secretKey);

    if (this.enabled) {
      this.client = new S3Client({
        region: config.storage.region || 'us-east-1',
        endpoint: config.storage.endpoint,
        forcePathStyle: Boolean(config.storage.endpoint),
        credentials: {
          accessKeyId: config.storage.accessKey as string,
          secretAccessKey: config.storage.secretKey as string,
        },
      });
    }
  }

  private generateKey(prefix: string, extension?: string): string {
    const random = crypto.randomBytes(12).toString('hex');
    const ext = extension ? `.${extension.replace(/^\./, '')}` : '';
    return `${prefix}/${Date.now()}-${random}${ext}`;
  }

  async uploadBuffer(
    buffer: Buffer,
    contentType: string,
    options: { prefix?: string; extension?: string } = {},
  ): Promise<string> {
    const key = this.generateKey(options.prefix || 'uploads', options.extension);

    if (!this.enabled || !this.client) {
      return key;
    }

    await this.client.send(
      new PutObjectCommand({
        Bucket: config.storage.bucket,
        Key: key,
        Body: buffer,
        ContentType: contentType,
      }),
    );

    return key;
  }

  async getSignedUrl(key: string, expiresInSeconds = 3600): Promise<string> {
    if (!this.enabled || !this.client) {
      const base = config.storage.endpoint || 'https://example.com/mock-storage';
      return `${base.replace(/\/$/, '')}/${key}`;
    }

    return getSignedUrl(
      this.client,
      new GetObjectCommand({ Bucket: config.storage.bucket, Key: key }),
      { expiresIn: expiresInSeconds },
    );
  }
}

const storageService = new StorageService();

export default storageService;
