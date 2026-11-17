import Link from "next/link";
import { prisma } from "@/lib/prisma";
import { DashboardShell } from "./components/dashboard-shell";
import { MetricsRow } from "./components/metrics-row";
import { ReportCard } from "./components/report-card";
import { CandidateTable } from "./components/candidate-table";
import { Sparkles, Video, Users, ArrowRight, Code2, ShieldCheck } from "lucide-react";

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
  const recentInterviews = interviews.slice(0, 5);
  const apiBase = process.env.NEXT_PUBLIC_API_BASE_URL ?? "http://localhost:8000/api";

  const completed = interviews.filter((i) => i.status === "completed");
  const totalScore = completed.reduce((acc, curr) => acc + (curr.finalScore ?? 0), 0);
  const avgScore = completed.length > 0 ? totalScore / completed.length : 0.82;

  return (
    <DashboardShell apiBaseUrl={apiBase}>
      {/* Page Title & Top CTA Header */}
      <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-4 mb-8">
        <div>
          <div className="flex items-center gap-2 mb-1">
            <h1 className="text-2xl sm:text-3xl font-extrabold tracking-tight text-gray-900 dark:text-white">
              Executive Overview
            </h1>
            <span className="px-2.5 py-0.5 rounded-full text-xs font-semibold bg-brand-50 text-brand-600 dark:bg-brand-500/15 dark:text-brand-400">
              Live Hub
            </span>
          </div>
          <p className="text-sm text-gray-500 dark:text-gray-400">
            Ringkasan metrik rekrutmen teknik, kecepatan bicara (WPM), dan evaluasi konsep React/Next.js.
          </p>
        </div>

        {/* Primary Action Button to /assessment */}
        <div className="flex items-center gap-3">
          <Link
            href="/assessment"
            className="inline-flex items-center gap-2 px-5 py-2.5 rounded-xl font-medium text-sm text-white bg-brand-500 hover:bg-brand-600 shadow-md shadow-brand-500/20 transition-all"
          >
            <Video className="w-4 h-4" />
            Mulai Asesmen Baru
            <ArrowRight className="w-4 h-4" />
          </Link>
        </div>
      </div>

      {/* Top 4 KPI Metrics Row */}
      <MetricsRow
        totalInterviews={interviews.length}
        averageScore={avgScore}
        completedCount={completed.length}
      />

      {/* Quick Overview Bento Grid */}
      <div className="grid grid-cols-12 gap-6 mb-8">
        {/* Welcome & Assessment Launcher Banner (Left 7 Cols) */}
        <div className="col-span-12 xl:col-span-7 tailadmin-card p-6 sm:p-7 flex flex-col justify-between bg-gradient-to-br from-white via-white to-brand-50/30 dark:from-[#101828] dark:via-[#101828] dark:to-brand-950/20">
          <div>
            <div className="flex items-center gap-2 text-brand-600 dark:text-brand-400 text-xs font-bold uppercase tracking-wider mb-2">
              <Sparkles className="w-4 h-4" />
              Sistem Asesmen Terotomasi
            </div>
            <h2 className="text-xl sm:text-2xl font-bold text-gray-900 dark:text-white mb-2">
              Evaluasi Kemampuan Frontend Engineer Secara Objektif
            </h2>
            <p className="text-sm text-gray-500 dark:text-gray-400 mb-6 leading-relaxed">
              Unggah video wawancara kandidat untuk dianalisis oleh pipeline Whisper STT dan NLP Scoring model. Sistem otomatis mengukur relevansi arsitektur React/Next.js, state management, dan tempo bicara (WPM).
            </p>

            <div className="grid grid-cols-1 sm:grid-cols-2 gap-3 mb-6">
              <div className="p-3.5 rounded-xl bg-gray-50 dark:bg-gray-900/60 border border-gray-100 dark:border-gray-800">
                <span className="text-xs font-bold text-gray-900 dark:text-white block mb-0.5">
                  🎯 Poin Kunci Otomatis
                </span>
                <span className="text-[11px] text-gray-400">
                  Component, App Router, Zustand, REST API
                </span>
              </div>
              <div className="p-3.5 rounded-xl bg-gray-50 dark:bg-gray-900/60 border border-gray-100 dark:border-gray-800">
                <span className="text-xs font-bold text-gray-900 dark:text-white block mb-0.5">
                  🎙️ Akustik Bebas Bias
                </span>
                <span className="text-[11px] text-gray-400">
                  Deteksi filler words & rentang ideal 110-165 WPM
                </span>
              </div>
            </div>
          </div>

          <div className="flex flex-wrap items-center gap-3 pt-4 border-t border-gray-100 dark:border-gray-800/80">
            <Link
              href="/assessment"
              className="inline-flex items-center gap-2 px-4 py-2 rounded-xl text-xs font-semibold text-white bg-brand-500 hover:bg-brand-600 transition-colors"
            >
              Buka Studio Asesmen Video
              <ArrowRight className="w-3.5 h-3.5" />
            </Link>
            <Link
              href="/rubrics"
              className="inline-flex items-center gap-1.5 px-4 py-2 rounded-xl text-xs font-semibold text-gray-700 dark:text-gray-300 bg-gray-100 dark:bg-gray-800 hover:bg-gray-200 dark:hover:bg-gray-700 transition-colors"
            >
              <Code2 className="w-3.5 h-3.5 text-brand-500" />
              Lihat Standar Rubrik
            </Link>
          </div>
        </div>

        {/* Latest Evaluation Preview (Right 5 Cols) */}
        <div className="col-span-12 xl:col-span-5 tailadmin-card p-6 sm:p-7">
          <div className="flex items-center justify-between pb-4 mb-5 border-b border-gray-100 dark:border-gray-800">
            <div className="flex items-center gap-2.5">
              <div className="p-2 rounded-xl bg-emerald-50 text-emerald-600 dark:bg-emerald-500/10 dark:text-emerald-400">
                <Sparkles className="w-5 h-5" />
              </div>
              <div>
                <h3 className="text-base font-bold text-gray-900 dark:text-white">
                  Evaluasi Terakhir
                </h3>
                <p className="text-xs text-gray-400">
                  Pratinjau skor sesi terbaru
                </p>
              </div>
            </div>
            <Link
              href="/candidates"
              className="text-xs font-semibold text-brand-600 dark:text-brand-400 hover:underline flex items-center gap-1"
            >
              Semua
              <ArrowRight className="w-3 h-3" />
            </Link>
          </div>
          <ReportCard interview={latestInterview} />
        </div>
      </div>

      {/* Recent Candidate Activity Table */}
      <section className="tailadmin-card p-6 sm:p-7">
        <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-3 pb-4 mb-5 border-b border-gray-100 dark:border-gray-800">
          <div className="flex items-center gap-2.5">
            <div className="p-2 rounded-xl bg-purple-50 text-purple-600 dark:bg-purple-500/10 dark:text-purple-400">
              <Users className="w-5 h-5" />
            </div>
            <div>
              <h3 className="text-base font-bold text-gray-900 dark:text-white">
                Sesi Wawancara Terbaru
              </h3>
              <p className="text-xs text-gray-400">
                5 kandidat terakhir yang dievaluasi sistem
              </p>
            </div>
          </div>
          <Link
            href="/candidates"
            className="inline-flex items-center gap-1.5 text-xs font-semibold text-brand-600 dark:text-brand-400 hover:underline"
          >
            Buka Direktori Lengkap ({interviews.length} data)
            <ArrowRight className="w-3.5 h-3.5" />
          </Link>
        </div>

        <CandidateTable interviews={recentInterviews} />
      </section>
    </DashboardShell>
  );
}
