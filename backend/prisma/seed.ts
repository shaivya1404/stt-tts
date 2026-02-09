import bcrypt from 'bcryptjs';
import { PrismaClient, UserRole } from '@prisma/client';

const prisma = new PrismaClient();

async function main(): Promise<void> {
  const orgName = process.env.SEED_ORG_NAME || 'Acme Corp';
  const orgSlug = process.env.SEED_ORG_SLUG || 'acme-corp';
  const adminEmail = process.env.SEED_ADMIN_EMAIL || 'owner@acme.dev';
  const adminPassword = process.env.SEED_ADMIN_PASSWORD || 'ChangeMe123!';

  const passwordHash = await bcrypt.hash(adminPassword, 10);

  const organization = await prisma.organization.upsert({
    where: { slug: orgSlug },
    update: { name: orgName },
    create: {
      name: orgName,
      slug: orgSlug,
    },
  });

  await prisma.user.upsert({
    where: {
      orgId_email: {
        orgId: organization.id,
        email: adminEmail,
      },
    },
    update: {
      role: UserRole.owner,
      passwordHash,
    },
    create: {
      orgId: organization.id,
      email: adminEmail,
      passwordHash,
      role: UserRole.owner,
    },
  });

  // Seed baseline ML models so management endpoints respond with data
  await prisma.mlModel.upsert({
    where: { name_type_version: { name: 'vits_multilingual', type: 'tts', version: '1.0.0' } },
    update: {},
    create: {
      name: 'vits_multilingual',
      type: 'tts',
      version: '1.0.0',
      status: 'active',
    },
  });

  await prisma.mlModel.upsert({
    where: { name_type_version: { name: 'conformer_rnnt', type: 'stt', version: '1.0.0' } },
    update: {},
    create: {
      name: 'conformer_rnnt',
      type: 'stt',
      version: '1.0.0',
      status: 'active',
    },
  });

  // eslint-disable-next-line no-console
  console.log('Seeded organization and admin user.');
}

main()
  .catch((err) => {
    // eslint-disable-next-line no-console
    console.error('Seed failed', err);
    process.exit(1);
  })
  .finally(async () => {
    await prisma.$disconnect();
  });
