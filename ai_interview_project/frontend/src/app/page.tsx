import { prisma } from "@/lib/prisma";
import { DashboardShell } from "./components/dashboard-shell";
import { MetricsRow } from "./components/metrics-row";
import { UploadPanel } from "./components/upload-panel";
import { ReportCard } from "./components/report-card";
import { CandidateTable } from "./components/candidate-table";
import { Sparkles, Video, Users, Code2, Activity } from "lucide-react";

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
    console.error("Prisma fetch error (fallback to empty list):", error);
    return [];
  }
}

export default async function DashboardPage() {
  const interviews = await getInterviews();
  const latestInterview = interviews[0] ?? null;
  const apiBase = process.env.NEXT_PUBLIC_API_BASE_URL ?? "http://localhost:8000/api";

  const completed = interviews.filter((i) => i.status === "completed");
  const totalScore = completed.reduce((acc, curr) => acc + (curr.finalScore ?? 0), 0);
  const avgScore = completed.length > 0 ? totalScore / completed.length : 0.82;

  return (
    <DashboardShell apiBaseUrl={apiBase}>
      {/* Page Title & Breadcrumb */}
      <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-4 mb-8">
        <div>
          <div className="flex items-center gap-2 mb-1">
            <h1 className="text-2xl sm:text-3xl font-extrabold tracking-tight text-gray-900 dark:text-white">
              AI Interview Analytics
            </h1>
            <span className="px-2.5 py-0.5 rounded-full text-xs font-semibold bg-brand-50 text-brand-600 dark:bg-brand-500/15 dark:text-brand-400">
              TailAdmin Style
            </span>
          </div>
          <p className="text-sm text-gray-500 dark:text-gray-400">
            Platform evaluasi wawancara otomatis dengan analisis Speech-to-Text Whisper dan scoring konsep NLP.
          </p>
        </div>

        {/* Quick Action Badges */}
        <div className="flex items-center gap-2">
          <div className="px-3.5 py-2 rounded-xl bg-white dark:bg-[#101828] border border-gray-200 dark:border-gray-800 text-xs font-medium text-gray-600 dark:text-gray-300 shadow-sm flex items-center gap-2">
            <span className="w-2 h-2 rounded-full bg-emerald-500" />
            Whisper Multilingual Active
          </div>
        </div>
      </div>

      {/* Top 4 KPI Metrics Row */}
      <MetricsRow
        totalInterviews={interviews.length}
        averageScore={avgScore}
        completedCount={completed.length}
      />

      {/* Bento Grid Layout (Upload & Ingestion vs Latest Report) */}
      <div className="grid grid-cols-12 gap-6 mb-8">
        {/* Upload Panel (Left 7 Cols) */}
        <section
          id="upload-section"
          className="col-span-12 xl:col-span-7 tailadmin-card p-6 sm:p-7 flex flex-col justify-between"
        >
          <div>
            <div className="flex items-center justify-between pb-4 mb-5 border-b border-gray-100 dark:border-gray-800">
              <div className="flex items-center gap-2.5">
                <div className="p-2 rounded-xl bg-brand-50 text-brand-600 dark:bg-brand-500/10 dark:text-brand-400">
                  <Video className="w-5 h-5" />
                </div>
                <div>
                  <h3 className="text-base font-bold text-gray-900 dark:text-white">
                    Video Assessment Studio
                  </h3>
                  <p className="text-xs text-gray-400">
                    Ekstraksi audio 16 kHz & transkripsi Whisper otomatis
                  </p>
                </div>
              </div>
              <span className="text-xs font-mono text-gray-400 bg-gray-50 dark:bg-gray-800/80 px-2 py-1 rounded-md">
                Step 1/2
              </span>
            </div>
            <UploadPanel apiBaseUrl={apiBase} />
          </div>
        </section>

        {/* Executive Evaluation Report (Right 5 Cols) */}
        <section
          id="analytics-section"
          className="col-span-12 xl:col-span-5 tailadmin-card p-6 sm:p-7"
        >
          <div className="flex items-center justify-between pb-4 mb-5 border-b border-gray-100 dark:border-gray-800">
            <div className="flex items-center gap-2.5">
              <div className="p-2 rounded-xl bg-emerald-50 text-emerald-600 dark:bg-emerald-500/10 dark:text-emerald-400">
                <Sparkles className="w-5 h-5" />
              </div>
              <div>
                <h3 className="text-base font-bold text-gray-900 dark:text-white">
                  Laporan Evaluasi Terakhir
                </h3>
                <p className="text-xs text-gray-400">
                  Ringkasan metrik verbal & checklist rubrik
                </p>
              </div>
            </div>
            <span className="text-xs font-mono text-gray-400 bg-gray-50 dark:bg-gray-800/80 px-2 py-1 rounded-md">
              Live Preview
            </span>
          </div>
          <ReportCard interview={latestInterview} />
        </section>
      </div>

      {/* Candidate History Table */}
      <section
        id="candidates-section"
        className="tailadmin-card p-6 sm:p-7 mb-8"
      >
        <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-3 pb-4 mb-5 border-b border-gray-100 dark:border-gray-800">
          <div className="flex items-center gap-2.5">
            <div className="p-2 rounded-xl bg-purple-50 text-purple-600 dark:bg-purple-500/10 dark:text-purple-400">
              <Users className="w-5 h-5" />
            </div>
            <div>
              <h3 className="text-base font-bold text-gray-900 dark:text-white">
                Candidate Directory & Interview Log
              </h3>
              <p className="text-xs text-gray-400">
                Data historis hasil evaluasi tersimpan via Prisma ORM
              </p>
            </div>
          </div>
          <span className="text-xs font-medium text-gray-500 dark:text-gray-400">
            Total {interviews.length} data tercatat
          </span>
        </div>

        <CandidateTable interviews={interviews} />
      </section>

      {/* React & Next.js Rubric Reference Section */}
      <section
        id="rubric-section"
        className="tailadmin-card p-6 sm:p-7 bg-gradient-to-r from-brand-50/50 via-white to-purple-50/30 dark:from-brand-950/20 dark:via-[#101828] dark:to-purple-950/20 border-brand-100 dark:border-brand-900/40"
      >
        <div className="flex items-center gap-3 mb-4">
          <div className="p-2 rounded-xl bg-brand-500 text-white shadow-sm shadow-brand-500/30">
            <Code2 className="w-5 h-5" />
          </div>
          <div>
            <h4 className="text-base font-bold text-gray-900 dark:text-white">
              Standar Rubrik Evaluasi: Frontend Engineer (React.js & Next.js)
            </h4>
            <p className="text-xs text-gray-500 dark:text-gray-400">
              Rubrik acuan otomatis yang diinjeksikan ke dalam scoring NLP dan STT initial_prompt.
            </p>
          </div>
        </div>

        <div className="grid grid-cols-1 md:grid-cols-3 gap-4 text-xs">
          <div className="p-4 rounded-xl bg-white dark:bg-[#101828] border border-gray-200 dark:border-gray-800 shadow-sm">
            <span className="font-bold text-brand-600 dark:text-brand-400 block mb-1">
              Junior Level (Threshold: 75%)
            </span>
            <p className="text-gray-600 dark:text-gray-300 mb-2">
              Menguasai React.js, Next.js, dan TypeScript, serta memahami arsitektur Component, Routing (App Router/Pages Router), State Management (Zustand/Context API), dan Data Fetching/API Integration.
            </p>
            <span className="text-[10px] text-gray-400 font-mono">Poin: Component · Routing · State · API</span>
          </div>

          <div className="p-4 rounded-xl bg-white dark:bg-[#101828] border border-gray-200 dark:border-gray-800 shadow-sm">
            <span className="font-bold text-emerald-600 dark:text-emerald-400 block mb-1">
              Mid Level (Threshold: 80%)
            </span>
            <p className="text-gray-600 dark:text-gray-300 mb-2">
              Arsitektur Componentisasi skalabel, optimasi performa render (memo, lazy loading, SSR/SSG/ISR), global State Management (Zustand/RTK), routing dinamis, dan data fetching handal.
            </p>
            <span className="text-[10px] text-gray-400 font-mono">Poin: SSR/SSG/ISR · Memoization · RTK</span>
          </div>

          <div className="p-4 rounded-xl bg-white dark:bg-[#101828] border border-gray-200 dark:border-gray-800 shadow-sm">
            <span className="font-bold text-purple-600 dark:text-purple-400 block mb-1">
              Senior Level (Threshold: 85%)
            </span>
            <p className="text-gray-600 dark:text-gray-300 mb-2">
              Design System, Micro-frontend atau monorepo, optimasi Core Web Vitals, observability & error tracking, strategi testing komprehensif, dan arsitektur state terdistribusi.
            </p>
            <span className="text-[10px] text-gray-400 font-mono">Poin: Core Web Vitals · Design System · CI/CD</span>
          </div>
        </div>
      </section>
    </DashboardShell>
  );
}
