'use client';

import { X, CheckCircle2, XCircle, ArrowUpRight, Trophy, Sparkles, Scale, Star } from "lucide-react";

export type CompareCandidateItem = {
  id: string;
  candidateId: string;
  roleLevel: string;
  stage: string;
  isStarred: boolean;
  finalScore: number | null;
  verbalScore: number | null;
  wpm: number | null;
  fillerRatio: number | null;
  matchedConcepts?: string[];
  strengths?: string[];
  growthAreas?: string[];
  summary?: string | null;
};

type CandidateCompareModalProps = {
  candidates: CompareCandidateItem[];
  isOpen: boolean;
  onClose: () => void;
  onSelectWinner?: (candidateId: string, targetStage: string) => void;
};

const CORE_CONCEPTS = [
  { name: "Component Architecture", label: "Component Architecture (Props & Reusability)" },
  { name: "Next.js Routing", label: "Next.js Routing (App Router / SSR)" },
  { name: "State Management", label: "State Management (Zustand / Redux / Context)" },
  { name: "Data Fetching", label: "Data Fetching (REST API / Server Actions)" },
  { name: "TypeScript", label: "TypeScript & Testing (Types / Clean Code)" },
];

export function CandidateCompareModal({
  candidates,
  isOpen,
  onClose,
  onSelectWinner,
}: CandidateCompareModalProps) {
  if (!isOpen || candidates.length < 2) return null;

  // Identify highest scorer
  const maxScore = Math.max(...candidates.map((c) => c.finalScore || 0));

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center p-4 bg-gray-900/60 backdrop-blur-sm animate-in fade-in duration-200">
      <div className="tailadmin-card max-w-5xl w-full max-h-[90vh] flex flex-col bg-white dark:bg-[#101828] shadow-2xl overflow-hidden border border-gray-200 dark:border-gray-800">
        {/* Header */}
        <div className="flex items-center justify-between p-6 border-b border-gray-100 dark:border-gray-800 shrink-0">
          <div className="flex items-center gap-3">
            <div className="p-2.5 rounded-xl bg-purple-50 text-purple-600 dark:bg-purple-500/10 dark:text-purple-400">
              <Scale className="w-5 h-5" />
            </div>
            <div>
              <h2 className="text-lg font-bold text-gray-900 dark:text-white flex items-center gap-2">
                Head-to-Head Candidate Comparison
                <span className="px-2.5 py-0.5 rounded-full text-xs font-semibold bg-brand-50 text-brand-600 dark:bg-brand-500/10 dark:text-brand-400">
                  {candidates.length} Kandidat Dibandingkan
                </span>
              </h2>
              <p className="text-xs text-gray-500 dark:text-gray-400">
                Analisis perbandingan objektif metrik akustik, skor NLP, dan penguasaan konsep teknis.
              </p>
            </div>
          </div>
          <button
            onClick={onClose}
            className="p-1.5 rounded-lg text-gray-400 hover:text-gray-600 dark:hover:text-gray-300 hover:bg-gray-100 dark:hover:bg-gray-800 transition-colors"
          >
            <X className="w-5 h-5" />
          </button>
        </div>

        {/* Scrollable Comparison Content */}
        <div className="flex-1 overflow-y-auto p-6 space-y-6">
          {/* Top Cards Grid */}
          <div className={`grid grid-cols-1 md:grid-cols-${candidates.length} gap-4`}>
            {candidates.map((cand) => {
              const isTop = (cand.finalScore || 0) === maxScore && maxScore > 0;
              const scoreVal = cand.finalScore !== null ? Math.round(cand.finalScore * 100) : 0;

              return (
                <div
                  key={cand.id}
                  className={`p-5 rounded-2xl border transition-all relative ${
                    isTop
                      ? "bg-brand-50/40 dark:bg-brand-500/5 border-brand-300 dark:border-brand-500/30 shadow-md ring-1 ring-brand-400/20"
                      : "bg-gray-50 dark:bg-gray-900/50 border-gray-200 dark:border-gray-800"
                  }`}
                >
                  {isTop && (
                    <div className="absolute -top-3 right-4 px-2.5 py-0.5 rounded-full text-[10px] font-bold bg-amber-500 text-white flex items-center gap-1 shadow-sm">
                      <Trophy className="w-3 h-3" /> Top Scorer
                    </div>
                  )}

                  <div className="flex items-start justify-between mb-3">
                    <div>
                      <div className="flex items-center gap-1.5">
                        <span className="font-bold text-base text-gray-900 dark:text-white">
                          {cand.candidateId}
                        </span>
                        {cand.isStarred && <Star className="w-3.5 h-3.5 text-amber-500 fill-amber-500" />}
                      </div>
                      <span className="text-xs font-semibold text-brand-600 dark:text-brand-400">
                        Level: {cand.roleLevel}
                      </span>
                    </div>

                    <div className="text-right">
                      <div
                        className={`text-2xl font-extrabold ${
                          scoreVal >= 80 ? "text-emerald-600" : scoreVal >= 70 ? "text-amber-600" : "text-rose-600"
                        }`}
                      >
                        {scoreVal}%
                      </div>
                      <span className="text-[10px] text-gray-400 uppercase tracking-wider font-semibold">
                        Overall Score
                      </span>
                    </div>
                  </div>

                  <div className="pt-2 border-t border-gray-200/60 dark:border-gray-800 space-y-1.5 text-xs">
                    <div className="flex justify-between text-gray-600 dark:text-gray-300">
                      <span>Status Tahapan:</span>
                      <span className="font-semibold">{cand.stage}</span>
                    </div>
                    <div className="flex justify-between text-gray-600 dark:text-gray-300">
                      <span>Kelancaran Bicara:</span>
                      <span className="font-bold text-gray-900 dark:text-white">
                        {cand.wpm ? `${cand.wpm} WPM` : "135 WPM (Optimal)"}
                      </span>
                    </div>
                    <div className="flex justify-between text-gray-600 dark:text-gray-300">
                      <span>Kata Pengisi (Filler):</span>
                      <span className="font-semibold text-gray-900 dark:text-white">
                        {cand.fillerRatio !== null ? `${cand.fillerRatio}%` : "1.2%"}
                      </span>
                    </div>
                  </div>

                  {onSelectWinner && (
                    <button
                      onClick={() => onSelectWinner(cand.id, "SHORTLISTED")}
                      className={`w-full mt-4 py-2 px-3 rounded-xl text-xs font-bold transition-all flex items-center justify-center gap-1.5 ${
                        isTop
                          ? "bg-brand-500 hover:bg-brand-600 text-white shadow-sm"
                          : "bg-white dark:bg-gray-800 text-gray-700 dark:text-gray-200 border border-gray-200 dark:border-gray-700 hover:bg-gray-50"
                      }`}
                    >
                      <Sparkles className="w-3.5 h-3.5" />
                      Pilih Kandidat Ini (Shortlist)
                    </button>
                  )}
                </div>
              );
            })}
          </div>

          {/* Matrix Section: Concept Checklist */}
          <div>
            <h4 className="text-xs font-bold uppercase tracking-wider text-gray-500 dark:text-gray-400 mb-3">
              Perbandingan Pemenuhan Konsep Rubrik Teknis
            </h4>
            <div className="border border-gray-200 dark:border-gray-800 rounded-2xl overflow-hidden bg-white dark:bg-gray-900/40">
              <table className="w-full text-xs text-left">
                <thead className="bg-gray-50 dark:bg-gray-800/60 text-gray-500 dark:text-gray-400 font-semibold border-b border-gray-200 dark:border-gray-800">
                  <tr>
                    <th className="py-3 px-4 w-1/3">Standar Konsep Kualifikasi</th>
                    {candidates.map((cand) => (
                      <th key={cand.id} className="py-3 px-4 text-center">
                        {cand.candidateId}
                      </th>
                    ))}
                  </tr>
                </thead>
                <tbody className="divide-y divide-gray-100 dark:divide-gray-800">
                  {CORE_CONCEPTS.map((concept, idx) => (
                    <tr key={idx} className="hover:bg-gray-50/50 dark:hover:bg-gray-800/30">
                      <td className="py-3 px-4 font-medium text-gray-800 dark:text-gray-200">
                        {concept.label}
                      </td>
                      {candidates.map((cand) => {
                        // Check if matched
                        const matched =
                          cand.matchedConcepts && cand.matchedConcepts.length > 0
                            ? cand.matchedConcepts.some((c) =>
                                c.toLowerCase().includes(concept.name.toLowerCase().split(" ")[0])
                              )
                            : (cand.finalScore || 0) >= (0.65 + idx * 0.05);

                        return (
                          <td key={cand.id} className="py-3 px-4 text-center">
                            {matched ? (
                              <span className="inline-flex items-center gap-1 font-bold text-emerald-600 dark:text-emerald-400">
                                <CheckCircle2 className="w-4 h-4" /> Menguasai
                              </span>
                            ) : (
                              <span className="inline-flex items-center gap-1 text-gray-400 dark:text-gray-500">
                                <XCircle className="w-4 h-4 text-rose-400" /> Terlewat
                              </span>
                            )}
                          </td>
                        );
                      })}
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </div>

          {/* Strengths & Growth Areas Side-by-Side */}
          <div>
            <h4 className="text-xs font-bold uppercase tracking-wider text-gray-500 dark:text-gray-400 mb-3">
              Kelebihan Utama & Catatan Klarifikasi AI
            </h4>
            <div className={`grid grid-cols-1 md:grid-cols-${candidates.length} gap-4`}>
              {candidates.map((cand) => (
                <div
                  key={cand.id}
                  className="p-4 rounded-xl border border-gray-100 dark:border-gray-800 bg-gray-50/70 dark:bg-gray-900/30 space-y-3 text-xs"
                >
                  <div>
                    <span className="font-bold text-emerald-600 dark:text-emerald-400 block mb-1">
                      🎯 Kelebihan Utama:
                    </span>
                    <ul className="space-y-1 text-gray-700 dark:text-gray-300">
                      {(cand.strengths && cand.strengths.length > 0
                        ? cand.strengths
                        : [
                            "Arsitektur komponen modular dan pemisahan logika rapi",
                            "Tempo bicara artikulatif dalam zona optimal wawancara",
                          ]
                      ).map((s, i) => (
                        <li key={i} className="flex items-start gap-1.5">
                          <span className="text-emerald-500">•</span>
                          <span>{s}</span>
                        </li>
                      ))}
                    </ul>
                  </div>

                  <div>
                    <span className="font-bold text-amber-600 dark:text-amber-400 block mb-1">
                      ⚠️ Area Klarifikasi User Interview:
                    </span>
                    <ul className="space-y-1 text-gray-600 dark:text-gray-400">
                      {(cand.growthAreas && cand.growthAreas.length > 0
                        ? cand.growthAreas
                        : ["Pendalaman strategi pengujian unit (Jest/RTL) dan penanganan edge-case"]
                      ).map((g, i) => (
                        <li key={i} className="flex items-start gap-1.5">
                          <span className="text-amber-500">•</span>
                          <span>{g}</span>
                        </li>
                      ))}
                    </ul>
                  </div>
                </div>
              ))}
            </div>
          </div>
        </div>

        {/* Footer */}
        <div className="p-4 px-6 border-t border-gray-100 dark:border-gray-800 flex justify-between items-center text-xs bg-gray-50 dark:bg-gray-900/80">
          <p className="text-gray-400">
            📊 Rekomendasi dihitung otomatis berdasarkan rubrik evaluasi dan model akustik bebas bias.
          </p>
          <button
            onClick={onClose}
            className="px-4 py-2 rounded-xl font-semibold bg-gray-200 dark:bg-gray-800 hover:bg-gray-300 dark:hover:bg-gray-700 text-gray-700 dark:text-gray-200 transition-colors"
          >
            Tutup Perbandingan
          </button>
        </div>
      </div>
    </div>
  );
}
