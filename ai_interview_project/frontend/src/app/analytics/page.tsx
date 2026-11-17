import { DashboardShell } from "../components/dashboard-shell";
import { Activity, Mic, Volume2, Sparkles, TrendingUp, Info, Clock, AlertCircle } from "lucide-react";

export const dynamic = "force-static";

export default function AnalyticsPage() {
  const apiBase = process.env.NEXT_PUBLIC_API_BASE_URL ?? "http://localhost:8000/api";

  return (
    <DashboardShell apiBaseUrl={apiBase}>
      {/* Header */}
      <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-4 mb-8">
        <div>
          <div className="flex items-center gap-2 mb-1">
            <h1 className="text-2xl sm:text-3xl font-extrabold tracking-tight text-gray-900 dark:text-white">
              Speech & NLP Metrics Analytics
            </h1>
            <span className="px-2.5 py-0.5 rounded-full text-xs font-semibold bg-brand-50 text-brand-600 dark:bg-brand-500/15 dark:text-brand-400">
              Acoustic Flow Engine
            </span>
          </div>
          <p className="text-sm text-gray-500 dark:text-gray-400">
            Penjelasan metrik kelancaran bicara tanpa bias bahasa, deteksi kata gumaman (*filler words*), dan scoring semantik.
          </p>
        </div>
      </div>

      {/* 3 Metric Cards Breakdown */}
      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6 mb-8">
        {/* Card 1: WPM Meter */}
        <div className="tailadmin-card p-6 sm:p-7 flex flex-col justify-between">
          <div>
            <div className="flex items-center gap-2.5 mb-4">
              <div className="p-2.5 rounded-xl bg-brand-50 text-brand-600 dark:bg-brand-500/10 dark:text-brand-400">
                <Clock className="w-5 h-5" />
              </div>
              <div>
                <h3 className="text-base font-bold text-gray-900 dark:text-white">
                  Speech Rate (WPM)
                </h3>
                <p className="text-xs text-gray-400">Words Per Minute</p>
              </div>
            </div>

            <div className="p-4 rounded-2xl bg-gray-50 dark:bg-gray-900/60 border border-gray-100 dark:border-gray-800 mb-4 text-center">
              <span className="text-3xl font-extrabold text-brand-600 dark:text-brand-400 block mb-1">
                110 – 165 WPM
              </span>
              <span className="text-xs font-semibold text-emerald-600 bg-emerald-50 dark:bg-emerald-500/15 px-2.5 py-0.5 rounded-full">
                Rentang Optimal Wawancara
              </span>
            </div>

            <div className="space-y-2 text-xs text-gray-600 dark:text-gray-300">
              <div className="flex justify-between py-1 border-b border-gray-100 dark:border-gray-800">
                <span>&lt; 80 WPM</span>
                <span className="font-semibold text-amber-500">Terlalu Lambat / Ragu-ragu</span>
              </div>
              <div className="flex justify-between py-1 border-b border-gray-100 dark:border-gray-800">
                <span>110 – 165 WPM</span>
                <span className="font-semibold text-emerald-500">Normal & Artikulatif</span>
              </div>
              <div className="flex justify-between py-1">
                <span>&gt; 210 WPM</span>
                <span className="font-semibold text-rose-500">Terlalu Cepat / Terburu-buru</span>
              </div>
            </div>
          </div>

          <p className="text-[11px] text-gray-400 pt-4 border-t border-gray-100 dark:border-gray-800/80 mt-4">
            Bobot kelancaran: 45% dari skor fluency total.
          </p>
        </div>

        {/* Card 2: Filler Words */}
        <div className="tailadmin-card p-6 sm:p-7 flex flex-col justify-between">
          <div>
            <div className="flex items-center gap-2.5 mb-4">
              <div className="p-2.5 rounded-xl bg-amber-50 text-amber-600 dark:bg-amber-500/10 dark:text-amber-400">
                <AlertCircle className="w-5 h-5" />
              </div>
              <div>
                <h3 className="text-base font-bold text-gray-900 dark:text-white">
                  Deteksi Filler Words
                </h3>
                <p className="text-xs text-gray-400">Kata Gumaman & Pengisi</p>
              </div>
            </div>

            <div className="p-4 rounded-2xl bg-gray-50 dark:bg-gray-900/60 border border-gray-100 dark:border-gray-800 mb-4 text-center">
              <span className="text-3xl font-extrabold text-amber-600 dark:text-amber-400 block mb-1">
                ≤ 2.0%
              </span>
              <span className="text-xs font-semibold text-emerald-600 bg-emerald-50 dark:bg-emerald-500/15 px-2.5 py-0.5 rounded-full">
                Batas Toleransi Ideal
              </span>
            </div>

            <div className="text-xs text-gray-600 dark:text-gray-300 space-y-2">
              <p className="font-semibold text-gray-900 dark:text-white">Kata yang dipantau:</p>
              <div className="flex flex-wrap gap-1.5">
                {["ehm", "uhm", "anu", "apa ya", "kayak", "gitu", "like", "you know"].map((f) => (
                  <span
                    key={f}
                    className="px-2 py-0.5 rounded-md text-[11px] bg-amber-50 dark:bg-amber-950/40 text-amber-700 dark:text-amber-400 border border-amber-200/50 dark:border-amber-800/50 font-mono"
                  >
                    &quot;{f}&quot;
                  </span>
                ))}
              </div>
            </div>
          </div>

          <p className="text-[11px] text-gray-400 pt-4 border-t border-gray-100 dark:border-gray-800/80 mt-4">
            Bobot penalti filler: 35% dari skor fluency total.
          </p>
        </div>

        {/* Card 3: Formula Bobot */}
        <div className="tailadmin-card p-6 sm:p-7 flex flex-col justify-between">
          <div>
            <div className="flex items-center gap-2.5 mb-4">
              <div className="p-2.5 rounded-xl bg-purple-50 text-purple-600 dark:bg-purple-500/10 dark:text-purple-400">
                <Sparkles className="w-5 h-5" />
              </div>
              <div>
                <h3 className="text-base font-bold text-gray-900 dark:text-white">
                  Formulasi Skor Verbal
                </h3>
                <p className="text-xs text-gray-400">Pembobotan Akhir</p>
              </div>
            </div>

            <div className="p-4 rounded-2xl bg-gray-50 dark:bg-gray-900/60 border border-gray-100 dark:border-gray-800 mb-4">
              <p className="text-xs font-mono text-brand-600 dark:text-brand-400 leading-relaxed font-bold">
                Verbal Score = (Relevance × 60%) + (Fluency × 40%)
              </p>
            </div>

            <div className="space-y-2 text-xs text-gray-600 dark:text-gray-300">
              <div className="p-2.5 rounded-xl bg-gray-50 dark:bg-gray-900/40 border border-gray-100 dark:border-gray-800">
                <strong className="text-gray-900 dark:text-white block">Relevance (60%):</strong>
                55% Checklist Konsep Kunci + 45% Kemiripan Semantik MiniLM.
              </div>
              <div className="p-2.5 rounded-xl bg-gray-50 dark:bg-gray-900/40 border border-gray-100 dark:border-gray-800">
                <strong className="text-gray-900 dark:text-white block">Fluency (40%):</strong>
                45% WPM Pace + 35% Filler Ratio + 20% Jeda Hening.
              </div>
            </div>
          </div>

          <p className="text-[11px] text-gray-400 pt-4 border-t border-gray-100 dark:border-gray-800/80 mt-4">
            Bebas dari bias model tata bahasa Inggris lama (CoLA).
          </p>
        </div>
      </div>
    </DashboardShell>
  );
}
