import Link from "next/link";
import { prisma } from "@/lib/prisma";

type InterviewRow = {
  id: string;
  candidateId: string;
  status: string;
  finalScore: number | null;
  verbalScore: number | null;
  nonVerbalScore: number | null;
  cheatingScore: number | null;
  updatedAt: Date;
  summary: string | null;
};

async function getInterviews(): Promise<InterviewRow[]> {
  return prisma.interview.findMany({
    orderBy: { updatedAt: "desc" },
    include: {
      transcript: false,
      nlp: false,
      vision: false,
    },
  });
}

export default async function DashboardPage() {
  const interviews = await getInterviews();

  return (
    <section className="space-y-8">
      <header className="space-y-2">
        <h2 className="text-2xl font-semibold text-slate-100">
          Interview Overview
        </h2>
        <p className="max-w-2xl text-sm text-slate-400">
          Data below is sourced from PostgreSQL via Prisma. Upload new sessions
          through the FastAPI backend, then refresh this dashboard to monitor
          scores, cheating indicators, and summaries.
        </p>
        <div className="flex items-center gap-3 text-xs text-slate-500">
          <span>Backend</span>
          <Link
            href="http://localhost:8000/docs"
            className="rounded bg-slate-800 px-2 py-1 font-medium text-slate-200 hover:bg-slate-700"
          >
            FastAPI Docs
          </Link>
          <span>API Base URL</span>
          <code className="rounded bg-slate-900 px-2 py-1">
            {process.env.NEXT_PUBLIC_API_BASE_URL ?? "http://localhost:8000/api"}
          </code>
        </div>
      </header>

      <div className="overflow-hidden rounded-xl border border-slate-800 bg-slate-900/40 shadow-lg">
        <table className="w-full border-collapse">
          <thead className="bg-slate-900/80 text-left text-xs uppercase tracking-wide text-slate-500">
            <tr>
              <th className="px-4 py-3">Candidate</th>
              <th className="px-4 py-3">Status</th>
              <th className="px-4 py-3">Final</th>
              <th className="px-4 py-3">Verbal</th>
              <th className="px-4 py-3">Non-Verbal</th>
              <th className="px-4 py-3">Cheating</th>
              <th className="px-4 py-3">Updated</th>
              <th className="px-4 py-3">Summary</th>
            </tr>
          </thead>
          <tbody className="text-sm">
            {interviews.length === 0 && (
              <tr>
                <td
                  colSpan={8}
                  className="px-4 py-6 text-center text-slate-500"
                >
                  No interviews stored yet. Trigger a processing job via the
                  backend API.
                </td>
              </tr>
            )}
            {interviews.map((row) => (
              <tr
                key={row.id}
                className="border-t border-slate-800/80 transition hover:bg-slate-800/40"
              >
                <td className="px-4 py-3 font-medium text-slate-100">
                  {row.candidateId}
                </td>
                <td className="px-4 py-3 capitalize text-slate-300">
                  {row.status}
                </td>
                <td className="px-4 py-3">
                  {row.finalScore?.toFixed(2) ?? "—"}
                </td>
                <td className="px-4 py-3">
                  {row.verbalScore?.toFixed(2) ?? "—"}
                </td>
                <td className="px-4 py-3">
                  {row.nonVerbalScore?.toFixed(2) ?? "—"}
                </td>
                <td className="px-4 py-3">
                  {row.cheatingScore?.toFixed(2) ?? "—"}
                </td>
                <td className="px-4 py-3 text-xs text-slate-400">
                  {row.updatedAt.toLocaleString()}
                </td>
                <td className="truncate px-4 py-3 text-xs text-slate-300">
                  {row.summary ?? "—"}
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </section>
  );
}
