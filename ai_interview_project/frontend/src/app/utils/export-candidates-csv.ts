/**
 * Utility to export candidate assessment data to CSV formatted for Microsoft Excel and Google Sheets.
 */

export type ExportCandidateRow = {
  candidateId: string;
  roleLevel: string;
  stage: string;
  finalScore: number | null;
  verbalScore: number | null;
  wpm: number | null;
  fillerRatio: number | null;
  isStarred: boolean;
  strengths: string[];
  growthAreas: string[];
  recruiterNotes: string | null;
  updatedAt: string | Date;
};

export function exportCandidatesToCSV(candidates: ExportCandidateRow[], filename = "kandidat-asesmen-ai.csv"): void {
  const headers = [
    "Candidate ID",
    "Level Kualifikasi",
    "Tahapan ATS (Stage)",
    "Skor Akhir (%)",
    "Skor Verbal (%)",
    "Kecepatan Bicara (WPM)",
    "Rasio Filler (%)",
    "Prioritas (Starred)",
    "Kelebihan Utama (AI)",
    "Area Pengembangan (AI)",
    "Catatan Rekruter",
    "Tanggal Update",
  ];

  const rows = candidates.map((c) => {
    const finalPct = c.finalScore !== null ? `${Math.round(c.finalScore * 100)}%` : "-";
    const verbalPct = c.verbalScore !== null ? `${Math.round(c.verbalScore * 100)}%` : "-";
    const wpmVal = c.wpm !== null ? `${c.wpm}` : "-";
    const fillerVal = c.fillerRatio !== null ? `${c.fillerRatio}%` : "-";
    const starredVal = c.isStarred ? "Ya ⭐" : "Tidak";
    const strengthsStr = c.strengths && c.strengths.length > 0 ? c.strengths.join(" | ") : "-";
    const growthStr = c.growthAreas && c.growthAreas.length > 0 ? c.growthAreas.join(" | ") : "-";
    const notesStr = c.recruiterNotes || "-";
    const dateStr = typeof c.updatedAt === "string" ? c.updatedAt : c.updatedAt.toISOString().split("T")[0];

    return [
      `"${c.candidateId}"`,
      `"${c.roleLevel}"`,
      `"${c.stage}"`,
      `"${finalPct}"`,
      `"${verbalPct}"`,
      `"${wpmVal}"`,
      `"${fillerVal}"`,
      `"${starredVal}"`,
      `"${strengthsStr.replace(/"/g, '""')}"`,
      `"${growthStr.replace(/"/g, '""')}"`,
      `"${notesStr.replace(/"/g, '""')}"`,
      `"${dateStr}"`,
    ].join(",");
  });

  // UTF-8 BOM to ensure Indonesian characters / accents render properly in Excel
  const csvContent = "\uFEFF" + [headers.join(","), ...rows].join("\r\n");
  const blob = new Blob([csvContent], { type: "text/csv;charset=utf-8;" });
  const url = URL.createObjectURL(blob);

  const link = document.createElement("a");
  link.setAttribute("href", url);
  link.setAttribute("download", filename);
  document.body.appendChild(link);
  link.click();
  document.body.removeChild(link);
  URL.revokeObjectURL(url);
}
