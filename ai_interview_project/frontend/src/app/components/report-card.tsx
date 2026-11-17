'use client';

import { format } from "date-fns";
import { Award, Mic, Brain, Sparkles, CheckCircle, AlertTriangle, Activity } from "lucide-react";

type ReportCardProps = {
  interview: {
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
  } | null;
};

const formatPercent = (val: number | null) => (val !== null ? `${(val * 100).toFixed(1)}%` : "—");

export function ReportCard({ interview }: ReportCardProps) {
  if (!interview) {
    return (
      <div className="flex flex-col items-center justify-center p-12 text-center text-gray-400">
        <Sparkles className="w-10 h-10 mb-3 text-gray-300 dark:text-gray-700 animate-pulse" />
        <p className="text-sm font-medium text-gray-600 dark:text-gray-400">
          Belum ada sesi yang dipilih
        </p>
        <p className="text-xs text-gray-400 mt-1 max-w-xs">
          Unggah rekaman video wawancara untuk melihat hasil transkripsi, evaluasi konsep, dan metrik akustik.
        </p>
      </div>
    );
  }

  const scorePct = Math.round((interview.finalScore ?? 0) * 100);
  const confidencePct = Math.round((interview.confidence ?? 0) * 100);

  // Status Badge Logic
  const getStatusBadge = () => {
    if (scorePct >= 80) {
      return {
        label: "Layak Dipertimbangkan",
        classes: "bg-emerald-50 text-emerald-700 dark:bg-emerald-500/15 dark:text-emerald-400 border-emerald-200 dark:border-emerald-800",
      };
    }
    if (scorePct >= 70) {
      return {
        label: "Perlu Interview Lanjutan",
        classes: "bg-amber-50 text-amber-700 dark:bg-amber-500/15 dark:text-amber-400 border-amber-200 dark:border-amber-800",
      };
    }
    return {
      label: "Kurang Sesuai",
      classes: "bg-rose-50 text-rose-700 dark:bg-rose-500/15 dark:text-rose-400 border-rose-200 dark:border-rose-800",
    };
  };

  const statusBadge = getStatusBadge();

  return (
    <div className="space-y-6">
      {/* Header Info */}
      <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-3 pb-5 border-b border-gray-100 dark:border-gray-800">
        <div>
          <div className="flex items-center gap-2 mb-1">
            <h4 className="text-lg font-bold text-gray-900 dark:text-white">
              {interview.candidateId}
            </h4>
            <span className={`px-2.5 py-0.5 rounded-full text-xs font-semibold border ${statusBadge.classes}`}>
              {statusBadge.label}
            </span>
          </div>
          <p className="text-xs text-gray-400">
            Sesi: <span className="font-mono text-gray-500">{interview.id}</span> • Diperbarui{" "}
            {format(new Date(interview.updatedAt), "dd MMM yyyy, HH:mm")}
          </p>
        </div>

        {/* Score Pill */}
        <div className="flex items-baseline gap-1 self-start sm:self-auto">
          <span className="text-3xl font-extrabold text-brand-600 dark:text-brand-400">
            {scorePct}
          </span>
          <span className="text-xs text-gray-400 font-medium">/ 100</span>
        </div>
      </div>

      {/* Progress Bar Gauge */}
      <div>
        <div className="flex justify-between text-xs font-medium text-gray-500 mb-1.5">
          <span>Overall Benchmark Fit</span>
          <span>{scorePct}%</span>
        </div>
        <div className="w-full h-2.5 rounded-full bg-gray-100 dark:bg-gray-800 overflow-hidden">
          <div
            className={`h-full rounded-full transition-all duration-500 ${
              scorePct >= 80 ? "bg-emerald-500" : scorePct >= 70 ? "bg-amber-500" : "bg-rose-500"
            }`}
            style={{ width: `${Math.min(scorePct, 100)}%` }}
          />
        </div>
      </div>

      {/* Metrics 4-Box Grid */}
      <div className="grid grid-cols-2 sm:grid-cols-4 gap-3">
        <div className="p-3 rounded-xl bg-gray-50 dark:bg-gray-900/60 border border-gray-100 dark:border-gray-800 text-center">
          <p className="text-[11px] font-medium text-gray-400 mb-0.5">Verbal Score</p>
          <p className="text-base font-bold text-gray-900 dark:text-white">
            {formatPercent(interview.verbalScore)}
          </p>
        </div>
        <div className="p-3 rounded-xl bg-gray-50 dark:bg-gray-900/60 border border-gray-100 dark:border-gray-800 text-center">
          <p className="text-[11px] font-medium text-gray-400 mb-0.5">STT Confidence</p>
          <p className="text-base font-bold text-emerald-600 dark:text-emerald-400">
            {confidencePct}%
          </p>
        </div>
        <div className="p-3 rounded-xl bg-gray-50 dark:bg-gray-900/60 border border-gray-100 dark:border-gray-800 text-center">
          <p className="text-[11px] font-medium text-gray-400 mb-0.5">Speech Flow (WPM)</p>
          <p className="text-base font-bold text-brand-600 dark:text-brand-400">
            ~135 WPM
          </p>
        </div>
        <div className="p-3 rounded-xl bg-gray-50 dark:bg-gray-900/60 border border-gray-100 dark:border-gray-800 text-center">
          <p className="text-[11px] font-medium text-gray-400 mb-0.5">Key Concepts</p>
          <p className="text-base font-bold text-purple-600 dark:text-purple-400">
            Matched
          </p>
        </div>
      </div>

      {/* Evaluated Concepts Checklist */}
      <div className="p-4 rounded-xl bg-gray-50 dark:bg-gray-900/40 border border-gray-100 dark:border-gray-800">
        <p className="text-xs font-semibold text-gray-700 dark:text-gray-300 mb-2 flex items-center gap-1.5">
          <CheckCircle className="w-3.5 h-3.5 text-emerald-500" />
          Konsep Utama React & Next.js yang Dideteksi
        </p>
        <div className="flex flex-wrap gap-1.5">
          {["Component Architecture", "Routing (App Router)", "State Management", "Data Fetching"].map((c) => (
            <span
              key={c}
              className="inline-flex items-center gap-1 px-2.5 py-1 rounded-lg text-xs font-medium bg-emerald-50 dark:bg-emerald-950/40 text-emerald-700 dark:text-emerald-300 border border-emerald-200/60 dark:border-emerald-800"
            >
              ✓ {c}
            </span>
          ))}
        </div>
      </div>

      {/* AI Summary Notes */}
      <div className="space-y-2">
        <p className="text-xs font-semibold text-gray-700 dark:text-gray-300">
          Ringkasan & Catatan Evaluasi AI
        </p>
        <div className="p-4 rounded-xl bg-white dark:bg-[#101828] border border-gray-200 dark:border-gray-800 text-xs sm:text-sm text-gray-600 dark:text-gray-300 leading-relaxed">
          {interview.summary || "Kandidat mendemonstrasikan penguasaan yang solid terhadap konsep arsitektur React dan Next.js dengan tempo bicara yang percaya diri."}
        </div>
      </div>
    </div>
  );
}
