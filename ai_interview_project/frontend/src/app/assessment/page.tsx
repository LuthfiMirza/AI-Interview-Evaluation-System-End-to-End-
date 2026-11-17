import { prisma } from "@/lib/prisma";
import { DashboardShell } from "../components/dashboard-shell";
import { UploadPanel } from "../components/upload-panel";
import { ReportCard } from "../components/report-card";
import { Video, Sparkles, CheckCircle2, Shield, Mic, Brain } from "lucide-react";

export const dynamic = "force-dynamic";

async function getLatestInterview() {
  try {
    return await prisma.interview.findFirst({
      orderBy: { updatedAt: "desc" },
    });
  } catch (error) {
    console.error("Prisma fetch error:", error);
    return null;
  }
}

export default async function AssessmentPage() {
  const latestInterview = await getLatestInterview();
  const apiBase = process.env.NEXT_PUBLIC_API_BASE_URL ?? "http://localhost:8000/api";

  return (
    <DashboardShell apiBaseUrl={apiBase}>
      {/* Page Header */}
      <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-4 mb-8">
        <div>
          <div className="flex items-center gap-2 mb-1">
            <h1 className="text-2xl sm:text-3xl font-extrabold tracking-tight text-gray-900 dark:text-white">
              Video Assessment Studio
            </h1>
            <span className="px-2.5 py-0.5 rounded-full text-xs font-semibold bg-brand-50 text-brand-600 dark:bg-brand-500/15 dark:text-brand-400">
              Interactive Ingestion
            </span>
          </div>
          <p className="text-sm text-gray-500 dark:text-gray-400">
            Unggah rekaman video kandidat dan evaluasi kualifikasi Frontend Engineer (React.js & Next.js) secara otomatis.
          </p>
        </div>
      </div>

      {/* Main Grid: Upload Studio on Left, Live Result on Right */}
      <div className="grid grid-cols-12 gap-6 mb-8">
        {/* Upload Form (Left 7 Cols) */}
        <section className="col-span-12 xl:col-span-7 tailadmin-card p-6 sm:p-7">
          <div className="flex items-center justify-between pb-4 mb-5 border-b border-gray-100 dark:border-gray-800">
            <div className="flex items-center gap-2.5">
              <div className="p-2 rounded-xl bg-brand-50 text-brand-600 dark:bg-brand-500/10 dark:text-brand-400">
                <Video className="w-5 h-5" />
              </div>
              <div>
                <h3 className="text-base font-bold text-gray-900 dark:text-white">
                  Formulir Penilaian Video
                </h3>
                <p className="text-xs text-gray-400">
                  Pilih level target dan lampirkan rekaman MP4 / WebM / MOV
                </p>
              </div>
            </div>
            <span className="text-xs font-semibold text-emerald-600 dark:text-emerald-400 bg-emerald-50 dark:bg-emerald-500/15 px-2.5 py-1 rounded-full">
              Pipeline Ready
            </span>
          </div>

          <UploadPanel apiBaseUrl={apiBase} />
        </section>

        {/* Live Evaluation Preview (Right 5 Cols) */}
        <section className="col-span-12 xl:col-span-5 tailadmin-card p-6 sm:p-7">
          <div className="flex items-center justify-between pb-4 mb-5 border-b border-gray-100 dark:border-gray-800">
            <div className="flex items-center gap-2.5">
              <div className="p-2 rounded-xl bg-emerald-50 text-emerald-600 dark:bg-emerald-500/10 dark:text-emerald-400">
                <Sparkles className="w-5 h-5" />
              </div>
              <div>
                <h3 className="text-base font-bold text-gray-900 dark:text-white">
                  Hasil Penilaian Terakhir
                </h3>
                <p className="text-xs text-gray-400">
                  Pratinjau metrik verbal & konsep
                </p>
              </div>
            </div>
          </div>

          <ReportCard interview={latestInterview} />
        </section>
      </div>

      {/* Pipeline Technical Process Overview */}
      <div className="tailadmin-card p-6 sm:p-7">
        <h4 className="text-sm font-bold text-gray-900 dark:text-white mb-4">
          Tahapan Pemrosesan Otomatis Pipeline AI
        </h4>
        <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4 text-xs">
          <div className="p-4 rounded-xl bg-gray-50 dark:bg-gray-900/60 border border-gray-100 dark:border-gray-800">
            <span className="w-6 h-6 rounded-full bg-brand-500 text-white flex items-center justify-center font-bold mb-2 text-[11px]">
              1
            </span>
            <p className="font-bold text-gray-900 dark:text-white mb-1">
              Audio Cleaning
            </p>
            <p className="text-gray-500 dark:text-gray-400">
              Ekstraksi audio via ffmpeg, filter biquad noise reduction, dan resampling 16 kHz mono.
            </p>
          </div>

          <div className="p-4 rounded-xl bg-gray-50 dark:bg-gray-900/60 border border-gray-100 dark:border-gray-800">
            <span className="w-6 h-6 rounded-full bg-brand-500 text-white flex items-center justify-center font-bold mb-2 text-[11px]">
              2
            </span>
            <p className="font-bold text-gray-900 dark:text-white mb-1">
              Whisper STT
            </p>
            <p className="text-gray-500 dark:text-gray-400">
              Transkripsi ucapan dengan panduan initial_prompt jargon React, Next.js, dan TypeScript.
            </p>
          </div>

          <div className="p-4 rounded-xl bg-gray-50 dark:bg-gray-900/60 border border-gray-100 dark:border-gray-800">
            <span className="w-6 h-6 rounded-full bg-brand-500 text-white flex items-center justify-center font-bold mb-2 text-[11px]">
              3
            </span>
            <p className="font-bold text-gray-900 dark:text-white mb-1">
              Acoustic & Rubric Scoring
            </p>
            <p className="text-gray-500 dark:text-gray-400">
              Perhitungan WPM (Speech Rate), deteksi filler words, jeda hening, dan checklist poin kunci.
            </p>
          </div>

          <div className="p-4 rounded-xl bg-gray-50 dark:bg-gray-900/60 border border-gray-100 dark:border-gray-800">
            <span className="w-6 h-6 rounded-full bg-brand-500 text-white flex items-center justify-center font-bold mb-2 text-[11px]">
              4
            </span>
            <p className="font-bold text-gray-900 dark:text-white mb-1">
              Database Sync
            </p>
            <p className="text-gray-500 dark:text-gray-400">
              Penyimpanan rekaman ke PostgreSQL / SQLite yang terhubung langsung ke dashboard via Prisma ORM.
            </p>
          </div>
        </div>
      </div>
    </DashboardShell>
  );
}
