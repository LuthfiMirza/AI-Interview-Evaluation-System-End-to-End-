'use client';

import { Star, Scale, Download, X, Layers, Check } from "lucide-react";

type FloatingBulkBarProps = {
  selectedCount: number;
  onClearSelection: () => void;
  onBulkStage: (stage: string) => void;
  onBulkStar: () => void;
  onBulkCompare: () => void;
  onBulkExport: () => void;
};

export function FloatingBulkBar({
  selectedCount,
  onClearSelection,
  onBulkStage,
  onBulkStar,
  onBulkCompare,
  onBulkExport,
}: FloatingBulkBarProps) {
  if (selectedCount === 0) return null;

  const canCompare = selectedCount === 2 || selectedCount === 3;

  return (
    <div className="fixed bottom-6 left-1/2 -translate-x-1/2 z-40 animate-in slide-in-from-bottom-5 duration-200">
      <div className="flex items-center gap-2 p-2.5 px-4 rounded-2xl bg-gray-900/95 dark:bg-gray-800/95 text-white shadow-2xl backdrop-blur-md border border-gray-700/60 text-xs">
        {/* Count Badge */}
        <div className="flex items-center gap-2 pr-3 border-r border-gray-700">
          <span className="w-5 h-5 rounded-full bg-brand-500 text-white font-bold flex items-center justify-center text-[10px]">
            {selectedCount}
          </span>
          <span className="font-semibold text-gray-200">Kandidat Terpilih</span>
        </div>

        {/* Action: Bulk Star */}
        <button
          onClick={onBulkStar}
          className="inline-flex items-center gap-1.5 py-1.5 px-3 rounded-xl bg-gray-800 hover:bg-gray-700 text-amber-400 font-semibold transition-colors"
          title="Tandai Bintang Semua"
        >
          <Star className="w-3.5 h-3.5 fill-amber-400" />
          Shortlist ⭐
        </button>

        {/* Action: Bulk Stage Select */}
        <div className="relative">
          <select
            onChange={(e) => {
              if (e.target.value) {
                onBulkStage(e.target.value);
                e.target.value = "";
              }
            }}
            defaultValue=""
            className="py-1.5 px-3 rounded-xl bg-gray-800 hover:bg-gray-700 text-gray-200 font-semibold focus:outline-none cursor-pointer border border-gray-700 text-xs"
          >
            <option value="" disabled>
              Ubah Status Tahapan...
            </option>
            <option value="INBOX">Inbox (Baru)</option>
            <option value="SHORTLISTED">Shortlisted ⭐</option>
            <option value="TECHNICAL_TEST">Technical Test</option>
            <option value="USER_INTERVIEW">User Interview</option>
            <option value="OFFERED">Offered</option>
            <option value="REJECTED">Rejected</option>
          </select>
        </div>

        {/* Action: Compare (2 or 3) */}
        <button
          onClick={onBulkCompare}
          disabled={!canCompare}
          className={`inline-flex items-center gap-1.5 py-1.5 px-3 rounded-xl font-semibold transition-all ${
            canCompare
              ? "bg-purple-600 hover:bg-purple-500 text-white shadow-sm"
              : "bg-gray-800/60 text-gray-500 cursor-not-allowed"
          }`}
          title={canCompare ? "Bandingkan kandidat terpilih" : "Pilih 2 atau 3 kandidat untuk membandingkan"}
        >
          <Scale className="w-3.5 h-3.5" />
          Bandingkan ({selectedCount})
        </button>

        {/* Action: Export Selected CSV */}
        <button
          onClick={onBulkExport}
          className="inline-flex items-center gap-1.5 py-1.5 px-3 rounded-xl bg-gray-800 hover:bg-gray-700 text-gray-200 font-semibold transition-colors"
          title="Download data kandidat terpilih ke CSV"
        >
          <Download className="w-3.5 h-3.5" />
          Export CSV
        </button>

        {/* Close / Cancel */}
        <button
          onClick={onClearSelection}
          className="p-1.5 rounded-lg text-gray-400 hover:text-white hover:bg-gray-700/80 transition-colors ml-1"
          title="Batal Pilihan"
        >
          <X className="w-4 h-4" />
        </button>
      </div>
    </div>
  );
}
