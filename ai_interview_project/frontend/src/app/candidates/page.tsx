import { prisma } from "@/lib/prisma";
import { DashboardShell } from "../components/dashboard-shell";
import { CandidateTable } from "../components/candidate-table";
import { Users, Filter, CheckCircle2, Clock, XCircle } from "lucide-react";

export const dynamic = "force-dynamic";

type InterviewRow = {
  id: string;
  candidateId: string;
  status: string;
  finalScore: number | null;
  verbalScore: number | null;
  nonVerbalScore: number | null;
  cheatingScore: number | null;
  confidence: number | null;
  summary: string | null;
  updatedAt: Date;
};

async function getInterviews(): Promise<InterviewRow[]> {
  try {
    return await prisma.interview.findMany({
      orderBy: { updatedAt: "desc" },
      select: {
        id: true,
        candidateId: true,
        status: true,
        finalScore: true,
        verbalScore: true,
        nonVerbalScore: true,
        cheatingScore: true,
        confidence: true,
        summary: true,
        updatedAt: true,
      },
    });
  } catch (error) {
    console.error("Prisma fetch error:", error);
    return [];
  }
}

export default async function CandidatesPage() {
  const interviews = await getInterviews();
  const apiBase = process.env.NEXT_PUBLIC_API_BASE_URL ?? "http://localhost:8000/api";

  const completed = interviews.filter((i) => i.status.toLowerCase() === "completed");
  const processing = interviews.filter((i) => i.status.toLowerCase() === "processing");
  const failed = interviews.filter((i) => i.status.toLowerCase() === "failed");

  return (
    <DashboardShell apiBaseUrl={apiBase}>
      {/* Header Title */}
      <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-4 mb-8">
        <div>
          <div className="flex items-center gap-2 mb-1">
            <h1 className="text-2xl sm:text-3xl font-extrabold tracking-tight text-gray-900 dark:text-white">
              Candidate Directory & Database
            </h1>
            <span className="px-2.5 py-0.5 rounded-full text-xs font-semibold bg-brand-50 text-brand-600 dark:bg-brand-500/15 dark:text-brand-400">
              Prisma Records
            </span>
          </div>
          <p className="text-sm text-gray-500 dark:text-gray-400">
            Daftar lengkap seluruh kandidat yang telah dievaluasi oleh sistem wawancara AI.
          </p>
        </div>
      </div>

      {/* Status Summary Pills Bar */}
      <div className="grid grid-cols-2 sm:grid-cols-4 gap-4 mb-8">
        <div className="tailadmin-card p-4 sm:p-5 flex items-center gap-3">
          <div className="p-2.5 rounded-xl bg-brand-50 text-brand-600 dark:bg-brand-500/10 dark:text-brand-400">
            <Users className="w-5 h-5" />
          </div>
          <div>
            <span className="text-xs text-gray-400 block font-medium">Total Kandidat</span>
            <span className="text-xl font-bold text-gray-900 dark:text-white">{interviews.length}</span>
          </div>
        </div>

        <div className="tailadmin-card p-4 sm:p-5 flex items-center gap-3">
          <div className="p-2.5 rounded-xl bg-emerald-50 text-emerald-600 dark:bg-emerald-500/10 dark:text-emerald-400">
            <CheckCircle2 className="w-5 h-5" />
          </div>
          <div>
            <span className="text-xs text-gray-400 block font-medium">Selesai (Completed)</span>
            <span className="text-xl font-bold text-emerald-600 dark:text-emerald-400">{completed.length}</span>
          </div>
        </div>

        <div className="tailadmin-card p-4 sm:p-5 flex items-center gap-3">
          <div className="p-2.5 rounded-xl bg-blue-50 text-blue-600 dark:bg-blue-500/10 dark:text-blue-400">
            <Clock className="w-5 h-5" />
          </div>
          <div>
            <span className="text-xs text-gray-400 block font-medium">Sedang Diproses</span>
            <span className="text-xl font-bold text-blue-600 dark:text-blue-400">{processing.length}</span>
          </div>
        </div>

        <div className="tailadmin-card p-4 sm:p-5 flex items-center gap-3">
          <div className="p-2.5 rounded-xl bg-rose-50 text-rose-600 dark:bg-rose-500/10 dark:text-rose-400">
            <XCircle className="w-5 h-5" />
          </div>
          <div>
            <span className="text-xs text-gray-400 block font-medium">Gagal Evaluasi</span>
            <span className="text-xl font-bold text-rose-600 dark:text-rose-400">{failed.length}</span>
          </div>
        </div>
      </div>

      {/* Main Full-Width Candidate Table Card */}
      <div className="tailadmin-card p-6 sm:p-7">
        <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-3 pb-4 mb-5 border-b border-gray-100 dark:border-gray-800">
          <div className="flex items-center gap-2">
            <Filter className="w-4 h-4 text-brand-500" />
            <h3 className="text-base font-bold text-gray-900 dark:text-white">
              Data Seluruh Kandidat
            </h3>
          </div>
          <span className="text-xs font-mono text-gray-400">
            Menampilkan {interviews.length} baris
          </span>
        </div>

        <CandidateTable interviews={interviews} />
      </div>
    </DashboardShell>
  );
}
