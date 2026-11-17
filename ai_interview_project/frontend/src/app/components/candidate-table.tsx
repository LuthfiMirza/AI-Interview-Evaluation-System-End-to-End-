'use client';

import { formatDistanceToNow } from "date-fns";
import { CheckCircle2, Clock, XCircle, ArrowUpRight } from "lucide-react";

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

type CandidateTableProps = {
  interviews: InterviewRow[];
  onSelectCandidate?: (id: string) => void;
};

export function CandidateTable({ interviews, onSelectCandidate }: CandidateTableProps) {
  const getStatusPill = (status: string) => {
    switch (status.toLowerCase()) {
      case "completed":
        return (
          <span className="inline-flex items-center gap-1.5 px-2.5 py-1 rounded-full text-xs font-semibold bg-emerald-50 text-emerald-700 dark:bg-emerald-500/15 dark:text-emerald-400 border border-emerald-200/60 dark:border-emerald-800">
            <CheckCircle2 className="w-3.5 h-3.5" />
            Completed
          </span>
        );
      case "failed":
        return (
          <span className="inline-flex items-center gap-1.5 px-2.5 py-1 rounded-full text-xs font-semibold bg-rose-50 text-rose-700 dark:bg-rose-500/15 dark:text-rose-400 border border-rose-200/60 dark:border-rose-800">
            <XCircle className="w-3.5 h-3.5" />
            Failed
          </span>
        );
      default:
        return (
          <span className="inline-flex items-center gap-1.5 px-2.5 py-1 rounded-full text-xs font-semibold bg-blue-50 text-blue-700 dark:bg-blue-500/15 dark:text-blue-400 border border-blue-200/60 dark:border-blue-800">
            <Clock className="w-3.5 h-3.5 animate-spin" />
            Processing
          </span>
        );
    }
  };

  const getScoreBadge = (score: number | null) => {
    if (score === null) return <span className="text-gray-400">—</span>;
    const val = Math.round(score * 100);
    const color =
      val >= 80
        ? "text-emerald-600 bg-emerald-50 dark:bg-emerald-500/10"
        : val >= 70
        ? "text-amber-600 bg-amber-50 dark:bg-amber-500/10"
        : "text-rose-600 bg-rose-50 dark:bg-rose-500/10";

    return (
      <span className={`px-2.5 py-1 rounded-lg text-xs font-bold ${color}`}>
        {val}%
      </span>
    );
  };

  return (
    <div className="overflow-x-auto">
      <table className="w-full text-left text-sm">
        <thead className="text-[11px] font-semibold text-gray-400 dark:text-gray-500 uppercase tracking-wider border-b border-gray-100 dark:border-gray-800">
          <tr>
            <th className="py-3.5 px-4">Candidate ID</th>
            <th className="py-3.5 px-4">Status</th>
            <th className="py-3.5 px-4">Overall Score</th>
            <th className="py-3.5 px-4">Verbal</th>
            <th className="py-3.5 px-4">Confidence</th>
            <th className="py-3.5 px-4">Updated</th>
            <th className="py-3.5 px-4">Ringkasan AI</th>
            <th className="py-3.5 px-4 text-right">Aksi</th>
          </tr>
        </thead>
        <tbody className="divide-y divide-gray-100 dark:divide-gray-800">
          {interviews.length === 0 ? (
            <tr>
              <td colSpan={8} className="py-12 text-center text-gray-400">
                Belum ada data wawancara yang tersimpan. Unggah video di panel atas untuk memulai evaluasi.
              </td>
            </tr>
          ) : (
            interviews.map((row) => (
              <tr
                key={row.id}
                className="hover:bg-gray-50/70 dark:hover:bg-gray-800/40 transition-colors group cursor-pointer"
                onClick={() => onSelectCandidate?.(row.id)}
              >
                {/* Candidate Name & Avatar */}
                <td className="py-4 px-4">
                  <div className="flex items-center gap-3">
                    <div className="w-8 h-8 rounded-full bg-gradient-to-tr from-brand-500 to-indigo-500 flex items-center justify-center text-white font-bold text-xs shrink-0">
                      {row.candidateId.slice(0, 2).toUpperCase()}
                    </div>
                    <div>
                      <span className="font-semibold text-gray-900 dark:text-white block group-hover:text-brand-500 transition-colors">
                        {row.candidateId}
                      </span>
                      <span className="text-[11px] text-gray-400 font-mono">
                        {row.id}
                      </span>
                    </div>
                  </div>
                </td>

                {/* Status */}
                <td className="py-4 px-4">{getStatusPill(row.status)}</td>

                {/* Scores */}
                <td className="py-4 px-4">{getScoreBadge(row.finalScore)}</td>
                <td className="py-4 px-4 font-medium text-gray-700 dark:text-gray-300">
                  {row.verbalScore !== null ? `${(row.verbalScore * 100).toFixed(1)}%` : "—"}
                </td>
                <td className="py-4 px-4 text-emerald-600 dark:text-emerald-400 font-medium">
                  {row.confidence !== null ? `${(row.confidence * 100).toFixed(0)}%` : "—"}
                </td>

                {/* Date */}
                <td className="py-4 px-4 text-xs text-gray-400 whitespace-nowrap">
                  {formatDistanceToNow(new Date(row.updatedAt), { addSuffix: true })}
                </td>

                {/* Summary */}
                <td className="py-4 px-4 max-w-xs">
                  <p className="truncate text-xs text-gray-500 dark:text-gray-400">
                    {row.summary || "—"}
                  </p>
                </td>

                {/* Action */}
                <td className="py-4 px-4 text-right">
                  <button
                    type="button"
                    className="inline-flex items-center gap-1 p-1.5 rounded-lg text-gray-400 hover:text-brand-500 hover:bg-brand-50 dark:hover:bg-brand-500/10 transition-colors"
                    title="Lihat Detail"
                  >
                    <ArrowUpRight className="w-4 h-4" />
                  </button>
                </td>
              </tr>
            ))
          )}
        </tbody>
      </table>
    </div>
  );
}
