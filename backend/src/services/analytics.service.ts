import prisma from '../lib/prisma';

const formatDay = (date: Date): string => date.toISOString().split('T')[0];

export const AnalyticsService = {
  async usage(orgId: string) {
    const now = new Date();
    const last30 = new Date(now.getTime() - 1000 * 60 * 60 * 24 * 30);
    const last7 = new Date(now.getTime() - 1000 * 60 * 60 * 24 * 7);

    const records = await prisma.usageRecord.findMany({
      where: {
        orgId,
        createdAt: {
          gte: last30,
        },
      },
      orderBy: { createdAt: 'asc' },
    });

    const daily = new Map<string, { date: string; tts: number; stt: number }>();
    let totalTts = 0;
    let totalStt = 0;

    records.forEach((record) => {
      const day = formatDay(record.createdAt);
      const entry = daily.get(day) || { date: day, tts: 0, stt: 0 };
      if (record.type === 'tts') {
        entry.tts += record.units;
        totalTts += record.units;
      } else {
        entry.stt += record.units;
        totalStt += record.units;
      }
      daily.set(day, entry);
    });

    const allDays = Array.from(daily.values()).sort((a, b) => (a.date > b.date ? 1 : -1));
    const last7Days = allDays.filter((day) => new Date(day.date) >= last7);

    return {
      totals: {
        totalTtsChars: totalTts,
        totalSttSeconds: totalStt,
      },
      last7Days,
      last30Days: allDays,
    };
  },
};
