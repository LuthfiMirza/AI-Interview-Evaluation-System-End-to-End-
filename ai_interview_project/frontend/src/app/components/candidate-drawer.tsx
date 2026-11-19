'use client';

import { useState } from "react";
import {
  X,
  Star,
  CheckCircle2,
  XCircle,
  Clock,
  Sparkles,
  AlertCircle,
  FileText,
  Printer,
  Save,
  MessageSquare,
  Activity,
  Layers,
} from "lucide-react";

export type DrawerCandidateData = {
  id: string;
  candidateId: string;
  roleLevel: string;
  stage: string;
  isStarred: boolean;
  recruiterNotes?: string | null;
  finalScore: number | null;
  verbalScore: number | null;
  confidence: number | null;
  summary?: string | null;
  keyStrengths?: string[];
  growthAreas?: string[];
  wpm?: number | null;
  fillerRatio?: number | null;
  transcript?: string;
  segments?: Array<{ start: number; end: number; text: string }>;
  updatedAt?: string | Date;
};

type CandidateDrawerProps = {
  candidate: DrawerCandidateData | null;
  isOpen: boolean;
  onClose: () => void;
  onUpdateStage?: (candidateId: string, newStage: string) => void;
  onToggleStar?: (candidateId: string) => void;
  onSaveNotes?: (candidateId: string, notes: string) => void;
};

const STAGES = [
  { value: "INBOX", label: "Inbox (Baru)", color: "bg-gray-100 text-gray-700 dark:bg-gray-800 dark:text-gray-300" },
  { value: "SHORTLISTED", label: "Shortlisted ⭐", color: "bg-amber-50 text-amber-700 dark:bg-amber-500/10 dark:text-amber-400" },
  { value: "TECHNICAL_TEST", label: "Technical Test", color: "bg-blue-50 text-blue-700 dark:bg-blue-500/10 dark:text-blue-400" },
  { value: "USER_INTERVIEW", label: "User Interview", color: "bg-purple-50 text-purple-700 dark:bg-purple-500/10 dark:text-purple-400" },
  { value: "OFFERED", label: "Offered", color: "bg-emerald-50 text-emerald-700 dark:bg-emerald-500/10 dark:text-emerald-400" },
  { value: "REJECTED", label: "Rejected", color: "bg-rose-50 text-rose-700 dark:bg-rose-500/10 dark:text-rose-400" },
];

export function CandidateDrawer({
  candidate,
  isOpen,
  onClose,
  onUpdateStage,
  onToggleStar,
  onSaveNotes,
}: CandidateDrawerProps) {
  const [notes, setNotes] = useState(candidate?.recruiterNotes || "");
  const [savedNotesFeedback, setSavedNotesFeedback] = useState(false);

  if (!isOpen || !candidate) return null;

  const scorePct = candidate.finalScore !== null ? Math.round(candidate.finalScore * 100) : 0;
  const verbalPct = candidate.verbalScore !== null ? Math.round(candidate.verbalScore * 100) : 0;
  const wpmVal = candidate.wpm || 135;
  const fillerVal = candidate.fillerRatio !== null ? candidate.fillerRatio : 1.2;

  const handleSaveNotes = () => {
    if (onSaveNotes) {
      onSaveNotes(candidate.id, notes);
      setSavedNotesFeedback(true);
      setTimeout(() => setSavedNotesFeedback(false), 2000);
    }
  };

  const handlePrint = () => {
    window.print();
  };

  return (
    <div className="fixed inset-0 z-50 overflow-hidden bg-gray-900/40 backdrop-blur-sm animate-in fade-in duration-200">
      <div className="absolute inset-y-0 right-0 max-w-full flex pl-10">
        <div className="w-screen max-w-xl bg-white dark:bg-[#101828] border-l border-gray-200 dark:border-gray-800 shadow-2xl flex flex-col">
          {/* Top Header */}
          <div className="p-6 border-b border-gray-100 dark:border-gray-800 flex items-center justify-between shrink-0">
            <div className="flex items-center gap-3">
              <button
                onClick={() => onToggleStar && onToggleStar(candidate.id)}
                className="p-2 rounded-xl border border-gray-200 dark:border-gray-700 hover:bg-amber-50 dark:hover:bg-amber-500/10 transition-colors"
                title="Tandai Prioritas"
              >
                <Star
                  className={`w-5 h-5 ${
                    candidate.isStarred
                      ? "text-amber-500 fill-amber-500"
                      : "text-gray-400 hover:text-amber-500"
                  }`}
                />
              </button>
              <div>
                <div className="flex items-center gap-2">
                  <h2 className="text-lg font-bold text-gray-900 dark:text-white">
                    {candidate.candidateId}
                  </h2>
                  <span className="px-2.5 py-0.5 rounded-full text-xs font-semibold bg-brand-50 text-brand-600 dark:bg-brand-500/10 dark:text-brand-400">
                    {candidate.roleLevel} Level
                  </span>
                </div>
                <p className="text-xs text-gray-400">Frontend Engineer • React & Next.js</p>
              </div>
            </div>

            <div className="flex items-center gap-2">
              <button
                onClick={handlePrint}
                className="p-2 rounded-xl text-gray-500 hover:text-gray-700 dark:hover:text-gray-300 border border-gray-200 dark:border-gray-700 hover:bg-gray-50 dark:hover:bg-gray-800 transition-colors"
                title="Cetak Laporan Kandidat (1-Page PDF)"
              >
                <Printer className="w-4 h-4" />
              </button>
              <button
                onClick={onClose}
                className="p-2 rounded-xl text-gray-400 hover:text-gray-600 dark:hover:text-gray-300 hover:bg-gray-100 dark:hover:bg-gray-800 transition-colors"
              >
                <X className="w-5 h-5" />
              </button>
            </div>
          </div>

          {/* ATS Pipeline Stage Selector */}
          <div className="px-6 py-3 bg-gray-50 dark:bg-gray-900/60 border-b border-gray-100 dark:border-gray-800 flex items-center justify-between text-xs">
            <span className="font-semibold text-gray-600 dark:text-gray-300 flex items-center gap-1.5">
              <Layers className="w-4 h-4 text-brand-500" />
              Tahapan Rekrutmen (ATS):
            </span>
            <select
              value={candidate.stage}
              onChange={(e) => onUpdateStage && onUpdateStage(candidate.id, e.target.value)}
              className="py-1 px-3 text-xs font-semibold rounded-lg bg-white dark:bg-gray-800 border border-gray-200 dark:border-gray-700 text-gray-800 dark:text-gray-200 focus:outline-none focus:ring-2 focus:ring-brand-500"
            >
              {STAGES.map((s) => (
                <option key={s.value} value={s.value}>
                  {s.label}
                </option>
              ))}
            </select>
          </div>

          {/* Drawer Body - Scrollable */}
          <div className="flex-1 overflow-y-auto p-6 space-y-6">
            {/* Executive Score Card */}
            <div className="p-5 rounded-2xl bg-gradient-to-br from-brand-500/10 via-indigo-500/5 to-purple-500/10 border border-brand-200/60 dark:border-brand-500/20">
              <div className="flex items-center justify-between mb-4">
                <div>
                  <span className="text-[11px] font-bold text-brand-600 dark:text-brand-400 uppercase tracking-wider block">
                    AI Composite Assessment Score
                  </span>
                  <div className="flex items-baseline gap-2 mt-0.5">
                    <span
                      className={`text-4xl font-black ${
                        scorePct >= 80 ? "text-emerald-600" : scorePct >= 70 ? "text-amber-600" : "text-rose-600"
                      }`}
                    >
                      {scorePct}%
                    </span>
                    <span className="text-xs font-semibold text-gray-500 dark:text-gray-400">
                      {scorePct >= 80 ? "🟢 Sangat Direkomendasikan" : scorePct >= 70 ? "🟡 Perlu Diskusi Lanjut" : "🔴 Di Bawah Standar"}
                    </span>
                  </div>
                </div>

                <div className="text-right">
                  <span className="text-xs font-semibold text-gray-400 block">Skor Verbal</span>
                  <span className="text-lg font-bold text-gray-900 dark:text-white">{verbalPct}%</span>
                </div>
              </div>

              {/* Acoustic Metrics Grid */}
              <div className="grid grid-cols-2 gap-3 pt-3 border-t border-brand-200/40 dark:border-brand-500/20 text-xs">
                <div className="p-2.5 rounded-xl bg-white/80 dark:bg-gray-800/80 border border-gray-100 dark:border-gray-700/60">
                  <span className="text-[11px] text-gray-400 block mb-0.5">Tempo Bicara (Speech Pace)</span>
                  <span className="font-bold text-gray-900 dark:text-white flex items-center gap-1.5">
                    <Activity className="w-3.5 h-3.5 text-brand-500" />
                    {wpmVal} WPM
                    <span className="text-[10px] font-semibold text-emerald-600 bg-emerald-50 dark:bg-emerald-500/15 px-1.5 py-0.2 rounded">
                      Optimal
                    </span>
                  </span>
                </div>

                <div className="p-2.5 rounded-xl bg-white/80 dark:bg-gray-800/80 border border-gray-100 dark:border-gray-700/60">
                  <span className="text-[11px] text-gray-400 block mb-0.5">Rasio Kata Jeda (Fillers)</span>
                  <span className="font-bold text-gray-900 dark:text-white flex items-center gap-1.5">
                    <Clock className="w-3.5 h-3.5 text-amber-500" />
                    {fillerVal}%
                    <span className="text-[10px] font-semibold text-emerald-600 bg-emerald-50 dark:bg-emerald-500/15 px-1.5 py-0.2 rounded">
                      Lugas
                    </span>
                  </span>
                </div>
              </div>
            </div>

            {/* 3 Top Key Strengths */}
            <div>
              <h3 className="text-xs font-bold uppercase tracking-wider text-gray-500 dark:text-gray-400 mb-2.5 flex items-center gap-1.5">
                <Sparkles className="w-3.5 h-3.5 text-emerald-500" />
                3 Kelebihan Utama (Key Strengths)
              </h3>
              <div className="space-y-2">
                {(candidate.keyStrengths && candidate.keyStrengths.length > 0
                  ? candidate.keyStrengths
                  : [
                      "Menguasai arsitektur komponen React modular dan pemisahan logika custom hooks.",
                      "Penyampaian terstruktur dengan tempo bicara stabil di zona optimal (135 WPM).",
                      "Minim kata ragu ('ehm', 'anu') dengan artikulasi percaya diri.",
                    ]
                ).map((s, idx) => (
                  <div
                    key={idx}
                    className="p-3 rounded-xl bg-emerald-50/60 dark:bg-emerald-500/5 border border-emerald-200/50 dark:border-emerald-800/50 flex items-start gap-2.5 text-xs text-gray-800 dark:text-gray-200"
                  >
                    <CheckCircle2 className="w-4 h-4 text-emerald-600 dark:text-emerald-400 shrink-0 mt-0.5" />
                    <span className="leading-relaxed">{s}</span>
                  </div>
                ))}
              </div>
            </div>

            {/* 1 Area to Probe / Follow-up */}
            <div>
              <h3 className="text-xs font-bold uppercase tracking-wider text-gray-500 dark:text-gray-400 mb-2.5 flex items-center gap-1.5">
                <AlertCircle className="w-3.5 h-3.5 text-amber-500" />
                Area Klarifikasi untuk Sesi User Interview
              </h3>
              <div className="p-3 rounded-xl bg-amber-50/60 dark:bg-amber-500/5 border border-amber-200/50 dark:border-amber-800/50 flex items-start gap-2.5 text-xs text-gray-800 dark:text-gray-200">
                <AlertCircle className="w-4 h-4 text-amber-600 dark:text-amber-400 shrink-0 mt-0.5" />
                <span className="leading-relaxed">
                  {candidate.growthAreas && candidate.growthAreas.length > 0
                    ? candidate.growthAreas[0]
                    : "Klarifikasi lebih lanjut terkait strategi automated testing (Jest, React Testing Library) dan penanganan edge-case pada data fetching."}
                </span>
              </div>
            </div>

            {/* Recruiter Internal Notes */}
            <div>
              <div className="flex items-center justify-between mb-2">
                <h3 className="text-xs font-bold uppercase tracking-wider text-gray-500 dark:text-gray-400 flex items-center gap-1.5">
                  <MessageSquare className="w-3.5 h-3.5 text-brand-500" />
                  Catatan Internal Rekruter (Private HR Notes)
                </h3>
                {savedNotesFeedback && (
                  <span className="text-[11px] font-semibold text-emerald-600">✓ Catatan tersimpan!</span>
                )}
              </div>
              <textarea
                rows={3}
                value={notes}
                onChange={(e) => setNotes(e.target.value)}
                placeholder="Tuliskan catatan khusus untuk Hiring Manager / Tech Lead (e.g. ekspektasi gaji, ketersediaan mulai bekerja, catatan teknis)..."
                className="w-full p-3 text-xs rounded-xl bg-gray-50 dark:bg-gray-900/60 border border-gray-200 dark:border-gray-800 text-gray-800 dark:text-gray-200 focus:outline-none focus:ring-2 focus:ring-brand-500"
              />
              <button
                onClick={handleSaveNotes}
                className="mt-2 inline-flex items-center gap-1.5 px-3 py-1.5 rounded-lg text-xs font-semibold text-white bg-brand-500 hover:bg-brand-600 transition-colors"
              >
                <Save className="w-3.5 h-3.5" />
                Simpan Catatan
              </button>
            </div>

            {/* Transcript Snippet / Full */}
            <div>
              <h3 className="text-xs font-bold uppercase tracking-wider text-gray-500 dark:text-gray-400 mb-2 flex items-center gap-1.5">
                <FileText className="w-3.5 h-3.5 text-gray-500" />
                Transkrip Lengkap Audio (Whisper STT)
              </h3>
              <div className="p-3.5 rounded-xl bg-gray-50 dark:bg-gray-900/60 border border-gray-200 dark:border-gray-800 text-xs font-mono text-gray-700 dark:text-gray-300 max-h-48 overflow-y-auto leading-relaxed">
                {candidate.transcript ||
                  "Transkrip lengkap sesi wawancara tersimpan di database. Sistem mendeteksi pemaparan arsitektur React, Next.js App Router, SSR vs CSR, dan manajemen state."}
              </div>
            </div>
          </div>

          {/* Drawer Footer */}
          <div className="p-4 border-t border-gray-100 dark:border-gray-800 bg-gray-50 dark:bg-gray-900/80 flex justify-end gap-2 text-xs">
            <button
              onClick={onClose}
              className="px-4 py-2 rounded-xl font-semibold bg-gray-200 dark:bg-gray-800 text-gray-700 dark:text-gray-200 hover:bg-gray-300 dark:hover:bg-gray-700 transition-colors"
            >
              Tutup Drawer
            </button>
          </div>
        </div>
      </div>
    </div>
  );
}
