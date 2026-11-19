'use client';

import { useState, useMemo } from "react";
import { formatDistanceToNow } from "date-fns";
import {
  CheckCircle2,
  Clock,
  XCircle,
  Star,
  Search,
  ArrowUpDown,
  Download,
  Mail,
  Eye,
  Scale,
  Sparkles,
  SlidersHorizontal,
  X,
  Layers,
  Activity,
  MessageSquare,
} from "lucide-react";
import { FloatingBulkBar } from "./floating-bulk-bar";
import { CandidateDrawer, DrawerCandidateData } from "./candidate-drawer";
import { CandidateCompareModal, CompareCandidateItem } from "./candidate-compare-modal";
import { RecruiterEmailModal, EmailCandidateData } from "./recruiter-email-modal";
import { WhatsAppNotificationModal, WhatsAppCandidateData } from "./whatsapp-notification-modal";
import { exportCandidatesToCSV, ExportCandidateRow } from "../utils/export-candidates-csv";

export type InterviewCandidate = {
  id: string;
  candidateId: string;
  status: string;
  stage?: string;
  isStarred?: boolean;
  roleLevel?: string;
  recruiterNotes?: string | null;
  finalScore: number | null;
  verbalScore: number | null;
  nonVerbalScore?: number | null;
  cheatingScore?: number | null;
  confidence: number | null;
  summary: string | null;
  keyStrengths?: string[];
  growthAreas?: string[];
  wpm?: number | null;
  fillerRatio?: number | null;
  transcript?: string;
  updatedAt: Date | string;
};

type CandidateTableProps = {
  interviews: InterviewCandidate[];
  apiBaseUrl?: string;
  onSelectCandidate?: (id: string) => void;
};

const STAGE_CONFIG: Record<string, { label: string; pill: string }> = {
  INBOX: {
    label: "Inbox",
    pill: "bg-gray-100 text-gray-700 dark:bg-gray-800 dark:text-gray-300 border-gray-200 dark:border-gray-700",
  },
  SHORTLISTED: {
    label: "Shortlisted ⭐",
    pill: "bg-amber-50 text-amber-700 dark:bg-amber-500/15 dark:text-amber-400 border-amber-200 dark:border-amber-800",
  },
  TECHNICAL_TEST: {
    label: "Tech Test",
    pill: "bg-blue-50 text-blue-700 dark:bg-blue-500/15 dark:text-blue-400 border-blue-200 dark:border-blue-800",
  },
  USER_INTERVIEW: {
    label: "User Interview",
    pill: "bg-purple-50 text-purple-700 dark:bg-purple-500/15 dark:text-purple-400 border-purple-200 dark:border-purple-800",
  },
  OFFERED: {
    label: "Offered",
    pill: "bg-emerald-50 text-emerald-700 dark:bg-emerald-500/15 dark:text-emerald-400 border-emerald-200 dark:border-emerald-800",
  },
  REJECTED: {
    label: "Rejected",
    pill: "bg-rose-50 text-rose-700 dark:bg-rose-500/15 dark:text-rose-400 border-rose-200 dark:border-rose-800",
  },
};

export function CandidateTable({ interviews: initialInterviews, apiBaseUrl }: CandidateTableProps) {
  const [candidates, setCandidates] = useState<InterviewCandidate[]>(initialInterviews);
  const [selectedIds, setSelectedIds] = useState<Set<string>>(new Set());

  // Search, Filter & Sort states
  const [searchQuery, setSearchQuery] = useState("");
  const [filterStage, setFilterStage] = useState<string>("ALL");
  const [filterLevel, setFilterLevel] = useState<string>("ALL");
  const [filterPassing, setFilterPassing] = useState<string>("ALL");
  const [starredOnly, setStarredOnly] = useState<boolean>(false);
  const [sortField, setSortField] = useState<string>("finalScore");
  const [sortOrder, setSortOrder] = useState<"asc" | "desc">("desc");

  // Modals state
  const [drawerCandidate, setDrawerCandidate] = useState<DrawerCandidateData | null>(null);
  const [emailCandidate, setEmailCandidate] = useState<EmailCandidateData | null>(null);
  const [whatsAppCandidate, setWhatsAppCandidate] = useState<WhatsAppCandidateData | null>(null);
  const [isCompareOpen, setIsCompareOpen] = useState(false);

  // Sync state if initialInterviews updates
  useMemo(() => {
    setCandidates(initialInterviews);
  }, [initialInterviews]);

  // Handle stage change (optimistic + API call)
  const handleUpdateStage = async (id: string, newStage: string) => {
    setCandidates((prev) =>
      prev.map((c) => (c.id === id ? { ...c, stage: newStage } : c))
    );
    if (drawerCandidate && drawerCandidate.id === id) {
      setDrawerCandidate({ ...drawerCandidate, stage: newStage });
    }

    if (apiBaseUrl) {
      try {
        await fetch(`${apiBaseUrl}/interviews/${id}/hr-metadata`, {
          method: "PATCH",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ stage: newStage }),
        });
      } catch (err) {
        console.error("Gagal update status stage:", err);
      }
    }
  };

  // Handle star toggle
  const handleToggleStar = async (id: string) => {
    const target = candidates.find((c) => c.id === id);
    const newStar = !target?.isStarred;

    setCandidates((prev) =>
      prev.map((c) => (c.id === id ? { ...c, isStarred: newStar } : c))
    );
    if (drawerCandidate && drawerCandidate.id === id) {
      setDrawerCandidate({ ...drawerCandidate, isStarred: newStar });
    }

    if (apiBaseUrl) {
      try {
        await fetch(`${apiBaseUrl}/interviews/${id}/hr-metadata`, {
          method: "PATCH",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ isStarred: newStar }),
        });
      } catch (err) {
        console.error("Gagal toggle star:", err);
      }
    }
  };

  // Handle save notes
  const handleSaveNotes = async (id: string, notes: string) => {
    setCandidates((prev) =>
      prev.map((c) => (c.id === id ? { ...c, recruiterNotes: notes } : c))
    );
    if (drawerCandidate && drawerCandidate.id === id) {
      setDrawerCandidate({ ...drawerCandidate, recruiterNotes: notes });
    }

    if (apiBaseUrl) {
      try {
        await fetch(`${apiBaseUrl}/interviews/${id}/hr-metadata`, {
          method: "PATCH",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ recruiterNotes: notes }),
        });
      } catch (err) {
        console.error("Gagal simpan notes:", err);
      }
    }
  };

  // Checkbox handlers
  const handleSelectAll = (checked: boolean) => {
    if (checked) {
      setSelectedIds(new Set(filteredCandidates.map((c) => c.id)));
    } else {
      setSelectedIds(new Set());
    }
  };

  const handleToggleRow = (id: string) => {
    const next = new Set(selectedIds);
    if (next.has(id)) {
      next.delete(id);
    } else {
      next.add(id);
    }
    setSelectedIds(next);
  };

  // Bulk actions
  const handleBulkStage = async (newStage: string) => {
    const ids = Array.from(selectedIds);
    setCandidates((prev) =>
      prev.map((c) => (selectedIds.has(c.id) ? { ...c, stage: newStage } : c))
    );
    setSelectedIds(new Set());

    if (apiBaseUrl) {
      try {
        await fetch(`${apiBaseUrl}/interviews/bulk`, {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({
            interview_ids: ids,
            action: "set_stage",
            stage: newStage,
          }),
        });
      } catch (err) {
        console.error("Gagal bulk stage:", err);
      }
    }
  };

  const handleBulkStar = async () => {
    const ids = Array.from(selectedIds);
    setCandidates((prev) =>
      prev.map((c) => (selectedIds.has(c.id) ? { ...c, isStarred: true, stage: "SHORTLISTED" } : c))
    );
    setSelectedIds(new Set());

    if (apiBaseUrl) {
      try {
        await fetch(`${apiBaseUrl}/interviews/bulk`, {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({
            interview_ids: ids,
            action: "set_star",
            is_starred: true,
            stage: "SHORTLISTED",
          }),
        });
      } catch (err) {
        console.error("Gagal bulk star:", err);
      }
    }
  };

  const handleBulkExport = () => {
    const targets = candidates.filter((c) => selectedIds.has(c.id));
    const exportData: ExportCandidateRow[] = targets.map((c) => ({
      candidateId: c.candidateId,
      roleLevel: c.roleLevel || "MID",
      stage: c.stage || "INBOX",
      finalScore: c.finalScore,
      verbalScore: c.verbalScore,
      wpm: c.wpm || 135,
      fillerRatio: c.fillerRatio || 1.2,
      isStarred: !!c.isStarred,
      strengths: c.keyStrengths || [],
      growthAreas: c.growthAreas || [],
      recruiterNotes: c.recruiterNotes || null,
      updatedAt: c.updatedAt,
    }));
    exportCandidatesToCSV(exportData, `kandidat-terpilih-${targets.length}.csv`);
  };

  const handleExportAllFiltered = () => {
    const exportData: ExportCandidateRow[] = filteredCandidates.map((c) => ({
      candidateId: c.candidateId,
      roleLevel: c.roleLevel || "MID",
      stage: c.stage || "INBOX",
      finalScore: c.finalScore,
      verbalScore: c.verbalScore,
      wpm: c.wpm || 135,
      fillerRatio: c.fillerRatio || 1.2,
      isStarred: !!c.isStarred,
      strengths: c.keyStrengths || [],
      growthAreas: c.growthAreas || [],
      recruiterNotes: c.recruiterNotes || null,
      updatedAt: c.updatedAt,
    }));
    exportCandidatesToCSV(exportData, `rekap-kandidat-asesmen-${filteredCandidates.length}.csv`);
  };

  // Filter & Sort calculation
  const filteredCandidates = useMemo(() => {
    return candidates
      .filter((c) => {
        // Star filter
        if (starredOnly && !c.isStarred) return false;

        // Stage filter
        if (filterStage !== "ALL" && (c.stage || "INBOX") !== filterStage) return false;

        // Level filter
        if (filterLevel !== "ALL" && (c.roleLevel || "MID").toUpperCase() !== filterLevel) return false;

        // Passing Grade filter
        if (filterPassing !== "ALL") {
          const score = (c.finalScore || 0) * 100;
          if (filterPassing === "PASS" && score < 80) return false;
          if (filterPassing === "BORDERLINE" && (score < 70 || score >= 80)) return false;
          if (filterPassing === "FAIL" && score >= 70) return false;
        }

        // Search Query (candidateId, level, summary, transcript)
        if (searchQuery.trim()) {
          const q = searchQuery.toLowerCase();
          const matchId = c.candidateId.toLowerCase().includes(q);
          const matchLevel = (c.roleLevel || "").toLowerCase().includes(q);
          const matchSummary = (c.summary || "").toLowerCase().includes(q);
          const matchTranscript = (c.transcript || "").toLowerCase().includes(q);
          if (!matchId && !matchLevel && !matchSummary && !matchTranscript) return false;
        }

        return true;
      })
      .sort((a, b) => {
        let valA: any = a.finalScore ?? -1;
        let valB: any = b.finalScore ?? -1;

        if (sortField === "wpm") {
          valA = a.wpm ?? 135;
          valB = b.wpm ?? 135;
        } else if (sortField === "verbalScore") {
          valA = a.verbalScore ?? -1;
          valB = b.verbalScore ?? -1;
        } else if (sortField === "updatedAt") {
          valA = new Date(a.updatedAt).getTime();
          valB = new Date(b.updatedAt).getTime();
        } else if (sortField === "candidateId") {
          valA = a.candidateId;
          valB = b.candidateId;
        }

        if (valA < valB) return sortOrder === "asc" ? -1 : 1;
        if (valA > valB) return sortOrder === "asc" ? 1 : -1;
        return 0;
      });
  }, [candidates, searchQuery, filterStage, filterLevel, filterPassing, starredOnly, sortField, sortOrder]);

  // Counts for summary pills
  const counts = useMemo(() => {
    return {
      total: candidates.length,
      starred: candidates.filter((c) => c.isStarred).length,
      pass: candidates.filter((c) => (c.finalScore || 0) >= 0.8).length,
      borderline: candidates.filter((c) => (c.finalScore || 0) >= 0.7 && (c.finalScore || 0) < 0.8).length,
      fail: candidates.filter((c) => (c.finalScore || 0) < 0.7).length,
    };
  }, [candidates]);

  // Candidates for Comparison Modal
  const compareList: CompareCandidateItem[] = useMemo(() => {
    return candidates
      .filter((c) => selectedIds.has(c.id))
      .map((c) => ({
        id: c.id,
        candidateId: c.candidateId,
        roleLevel: c.roleLevel || "MID",
        stage: c.stage || "INBOX",
        isStarred: !!c.isStarred,
        finalScore: c.finalScore,
        verbalScore: c.verbalScore,
        wpm: c.wpm || 135,
        fillerRatio: c.fillerRatio || 1.2,
        strengths: c.keyStrengths,
        growthAreas: c.growthAreas,
        summary: c.summary,
      }));
  }, [candidates, selectedIds]);

  const allFilteredSelected =
    filteredCandidates.length > 0 &&
    filteredCandidates.every((c) => selectedIds.has(c.id));

  return (
    <div className="space-y-4">
      {/* Top Filter & Search Control Bar */}
      <div className="tailadmin-card p-4 sm:p-5 space-y-4">
        {/* Row 1: Search & Sorting */}
        <div className="flex flex-col md:flex-row gap-3 items-stretch md:items-center justify-between">
          {/* Search Box */}
          <div className="relative flex-1">
            <Search className="w-4 h-4 text-gray-400 absolute left-3.5 top-1/2 -translate-y-1/2" />
            <input
              type="text"
              value={searchQuery}
              onChange={(e) => setSearchQuery(e.target.value)}
              placeholder="Cari candidate ID, level, catatan, atau istilah transkrip (e.g. 'Zustand', 'App Router')..."
              className="w-full pl-9 pr-8 py-2 text-xs rounded-xl bg-gray-50 dark:bg-gray-900/60 border border-gray-200 dark:border-gray-800 text-gray-900 dark:text-white placeholder-gray-400 focus:outline-none focus:ring-2 focus:ring-brand-500"
            />
            {searchQuery && (
              <button
                onClick={() => setSearchQuery("")}
                className="absolute right-3 top-1/2 -translate-y-1/2 text-gray-400 hover:text-gray-600 dark:hover:text-gray-200"
              >
                <X className="w-3.5 h-3.5" />
              </button>
            )}
          </div>

          {/* Sort Dropdown */}
          <div className="flex items-center gap-2">
            <div className="flex items-center gap-1.5 px-3 py-2 rounded-xl bg-gray-50 dark:bg-gray-900/60 border border-gray-200 dark:border-gray-800 text-xs">
              <ArrowUpDown className="w-3.5 h-3.5 text-gray-400" />
              <span className="text-gray-500 font-medium">Urutkan:</span>
              <select
                value={`${sortField}-${sortOrder}`}
                onChange={(e) => {
                  const [field, order] = e.target.value.split("-");
                  setSortField(field);
                  setSortOrder(order as "asc" | "desc");
                }}
                className="bg-transparent font-semibold text-gray-800 dark:text-gray-200 focus:outline-none cursor-pointer"
              >
                <option value="finalScore-desc">Skor Tertinggi (High-to-Low)</option>
                <option value="finalScore-asc">Skor Terendah (Low-to-High)</option>
                <option value="wpm-desc">Tempo Bicara WPM (Cepat)</option>
                <option value="wpm-asc">Tempo Bicara WPM (Tenang)</option>
                <option value="updatedAt-desc">Waktu Terbaru</option>
                <option value="candidateId-asc">Nama / ID (A-Z)</option>
              </select>
            </div>

            {/* Export CSV Button */}
            <button
              onClick={handleExportAllFiltered}
              className="inline-flex items-center gap-1.5 px-3 py-2 rounded-xl text-xs font-semibold bg-white dark:bg-gray-800 border border-gray-200 dark:border-gray-700 hover:bg-gray-50 dark:hover:bg-gray-700/60 text-gray-700 dark:text-gray-200 transition-colors shadow-sm"
              title="Download daftar kandidat saat ini ke format CSV (Excel)"
            >
              <Download className="w-3.5 h-3.5" />
              <span className="hidden sm:inline">Export Excel</span>
            </button>
          </div>
        </div>

        {/* Row 2: Filter Chips & Level Tabs */}
        <div className="flex flex-wrap items-center justify-between gap-3 pt-3 border-t border-gray-100 dark:border-gray-800 text-xs">
          {/* Quick Filters */}
          <div className="flex flex-wrap items-center gap-1.5">
            <button
              onClick={() => {
                setStarredOnly(false);
                setFilterPassing("ALL");
                setFilterStage("ALL");
              }}
              className={`px-3 py-1.5 rounded-xl font-semibold transition-all ${
                !starredOnly && filterPassing === "ALL" && filterStage === "ALL"
                  ? "bg-brand-500 text-white shadow-sm"
                  : "bg-gray-100 dark:bg-gray-800 text-gray-600 dark:text-gray-300 hover:bg-gray-200"
              }`}
            >
              Semua ({counts.total})
            </button>

            <button
              onClick={() => setStarredOnly(!starredOnly)}
              className={`inline-flex items-center gap-1 px-3 py-1.5 rounded-xl font-semibold transition-all ${
                starredOnly
                  ? "bg-amber-500 text-white shadow-sm"
                  : "bg-gray-100 dark:bg-gray-800 text-amber-600 dark:text-amber-400 hover:bg-amber-50"
              }`}
            >
              <Star className="w-3 h-3 fill-current" />
              Shortlisted ⭐ ({counts.starred})
            </button>

            <button
              onClick={() => setFilterPassing(filterPassing === "PASS" ? "ALL" : "PASS")}
              className={`px-3 py-1.5 rounded-xl font-semibold transition-all ${
                filterPassing === "PASS"
                  ? "bg-emerald-600 text-white shadow-sm"
                  : "bg-gray-100 dark:bg-gray-800 text-emerald-600 dark:text-emerald-400 hover:bg-emerald-50"
              }`}
            >
              🟢 Lolos ≥ 80% ({counts.pass})
            </button>

            <button
              onClick={() => setFilterPassing(filterPassing === "BORDERLINE" ? "ALL" : "BORDERLINE")}
              className={`px-3 py-1.5 rounded-xl font-semibold transition-all ${
                filterPassing === "BORDERLINE"
                  ? "bg-amber-600 text-white shadow-sm"
                  : "bg-gray-100 dark:bg-gray-800 text-amber-600 dark:text-amber-400 hover:bg-amber-50"
              }`}
            >
              🟡 Borderline 70-79% ({counts.borderline})
            </button>

            <button
              onClick={() => setFilterPassing(filterPassing === "FAIL" ? "ALL" : "FAIL")}
              className={`px-3 py-1.5 rounded-xl font-semibold transition-all ${
                filterPassing === "FAIL"
                  ? "bg-rose-600 text-white shadow-sm"
                  : "bg-gray-100 dark:bg-gray-800 text-rose-600 dark:text-rose-400 hover:bg-rose-50"
              }`}
            >
              🔴 Perlu Review &lt;70% ({counts.fail})
            </button>
          </div>

          {/* Level Tabs */}
          <div className="flex items-center gap-1 bg-gray-100 dark:bg-gray-800/80 p-1 rounded-xl">
            {["ALL", "JUNIOR", "MID", "SENIOR"].map((lvl) => (
              <button
                key={lvl}
                onClick={() => setFilterLevel(lvl)}
                className={`py-1 px-2.5 rounded-lg font-bold text-[11px] transition-all ${
                  filterLevel === lvl
                    ? "bg-white dark:bg-gray-700 text-brand-600 dark:text-brand-400 shadow-sm"
                    : "text-gray-500 hover:text-gray-900 dark:text-gray-400"
                }`}
              >
                {lvl === "ALL" ? "Semua Level" : lvl}
              </button>
            ))}
          </div>
        </div>
      </div>

      {/* Main Candidates Table */}
      <div className="tailadmin-card overflow-hidden">
        <div className="overflow-x-auto">
          <table className="w-full text-left text-xs">
            <thead className="text-[11px] font-semibold text-gray-400 dark:text-gray-500 uppercase tracking-wider bg-gray-50/70 dark:bg-gray-900/40 border-b border-gray-100 dark:border-gray-800">
              <tr>
                <th className="py-3 px-3 w-8">
                  <input
                    type="checkbox"
                    checked={allFilteredSelected}
                    onChange={(e) => handleSelectAll(e.target.checked)}
                    className="w-3.5 h-3.5 rounded text-brand-600 focus:ring-brand-500 cursor-pointer"
                  />
                </th>
                <th className="py-3 px-2 w-8 text-center">⭐</th>
                <th className="py-3 px-3">Candidate ID</th>
                <th className="py-3 px-3">Status Tahapan (ATS)</th>
                <th className="py-3 px-3 text-center">Skor Akhir</th>
                <th className="py-3 px-3">Kelancaran (WPM)</th>
                <th className="py-3 px-3">Kata Jeda</th>
                <th className="py-3 px-3">Poin Kelebihan AI</th>
                <th className="py-3 px-3 text-right">Aksi HR</th>
              </tr>
            </thead>

            <tbody className="divide-y divide-gray-100 dark:divide-gray-800">
              {filteredCandidates.length === 0 ? (
                <tr>
                  <td colSpan={9} className="py-12 text-center text-gray-400">
                    <div className="flex flex-col items-center justify-center gap-2">
                      <Search className="w-6 h-6 text-gray-300 dark:text-gray-600" />
                      <p className="font-semibold text-sm text-gray-600 dark:text-gray-300">
                        Tidak ada kandidat yang cocok dengan kriteria filter/pencarian.
                      </p>
                      <button
                        onClick={() => {
                          setSearchQuery("");
                          setFilterPassing("ALL");
                          setFilterLevel("ALL");
                          setFilterStage("ALL");
                          setStarredOnly(false);
                        }}
                        className="text-xs text-brand-600 dark:text-brand-400 font-semibold hover:underline"
                      >
                        Reset semua filter
                      </button>
                    </div>
                  </td>
                </tr>
              ) : (
                filteredCandidates.map((c) => {
                  const isChecked = selectedIds.has(c.id);
                  const scoreVal = c.finalScore !== null ? Math.round(c.finalScore * 100) : null;
                  const currentStage = c.stage || "INBOX";
                  const stageStyle = STAGE_CONFIG[currentStage] || STAGE_CONFIG.INBOX;
                  const wpm = c.wpm || 135;
                  const filler = c.fillerRatio !== null ? c.fillerRatio : 1.2;

                  return (
                    <tr
                      key={c.id}
                      className={`hover:bg-gray-50/60 dark:hover:bg-gray-800/40 transition-colors ${
                        isChecked ? "bg-brand-50/30 dark:bg-brand-500/5" : ""
                      }`}
                    >
                      {/* Checkbox */}
                      <td className="py-3 px-3">
                        <input
                          type="checkbox"
                          checked={isChecked}
                          onChange={() => handleToggleRow(c.id)}
                          className="w-3.5 h-3.5 rounded text-brand-600 focus:ring-brand-500 cursor-pointer"
                        />
                      </td>

                      {/* Star Button */}
                      <td className="py-3 px-2 text-center">
                        <button
                          onClick={() => handleToggleStar(c.id)}
                          className="p-1 rounded-lg hover:bg-gray-100 dark:hover:bg-gray-800 text-gray-300 hover:text-amber-500 transition-colors"
                          title="Tandai Prioritas"
                        >
                          <Star
                            className={`w-4 h-4 ${
                              c.isStarred
                                ? "text-amber-500 fill-amber-500"
                                : "text-gray-300 dark:text-gray-600"
                            }`}
                          />
                        </button>
                      </td>

                      {/* Candidate ID & Level */}
                      <td className="py-3 px-3">
                        <div className="flex items-center gap-2">
                          <button
                            onClick={() =>
                              setDrawerCandidate({
                                ...c,
                                roleLevel: c.roleLevel || "MID",
                                stage: currentStage,
                                isStarred: !!c.isStarred,
                                wpm,
                                fillerRatio: filler,
                              })
                            }
                            className="font-bold text-gray-900 dark:text-white hover:text-brand-600 dark:hover:text-brand-400 hover:underline text-left"
                          >
                            {c.candidateId}
                          </button>
                          <span className="px-2 py-0.5 rounded-full text-[10px] font-semibold bg-gray-100 dark:bg-gray-800 text-gray-600 dark:text-gray-300">
                            {c.roleLevel || "MID"}
                          </span>
                        </div>
                        <span className="text-[10px] text-gray-400 block mt-0.5">
                          {c.updatedAt ? formatDistanceToNow(new Date(c.updatedAt), { addSuffix: true }) : "Baru"}
                        </span>
                      </td>

                      {/* ATS Pipeline Stage Selector */}
                      <td className="py-3 px-3">
                        <select
                          value={currentStage}
                          onChange={(e) => handleUpdateStage(c.id, e.target.value)}
                          className={`py-1 px-2.5 rounded-full text-[11px] font-bold border focus:outline-none cursor-pointer ${stageStyle.pill}`}
                        >
                          <option value="INBOX">Inbox</option>
                          <option value="SHORTLISTED">Shortlisted ⭐</option>
                          <option value="TECHNICAL_TEST">Tech Test</option>
                          <option value="USER_INTERVIEW">User Interview</option>
                          <option value="OFFERED">Offered</option>
                          <option value="REJECTED">Rejected</option>
                        </select>
                      </td>

                      {/* Overall Score */}
                      <td className="py-3 px-3 text-center">
                        {scoreVal !== null ? (
                          <span
                            className={`inline-block font-extrabold px-2.5 py-1 rounded-lg text-xs ${
                              scoreVal >= 80
                                ? "text-emerald-700 bg-emerald-50 dark:bg-emerald-500/15 dark:text-emerald-400"
                                : scoreVal >= 70
                                ? "text-amber-700 bg-amber-50 dark:bg-amber-500/15 dark:text-amber-400"
                                : "text-rose-700 bg-rose-50 dark:bg-rose-500/15 dark:text-rose-400"
                            }`}
                          >
                            {scoreVal}%
                          </span>
                        ) : (
                          <span className="text-gray-400">—</span>
                        )}
                      </td>

                      {/* Speech Rate (WPM) */}
                      <td className="py-3 px-3">
                        <span className="font-semibold text-gray-800 dark:text-gray-200">
                          {wpm} WPM
                        </span>
                        <span className="text-[10px] text-emerald-600 dark:text-emerald-400 ml-1.5 font-bold">
                          Optimal
                        </span>
                      </td>

                      {/* Filler Ratio */}
                      <td className="py-3 px-3">
                        <span className="font-semibold text-gray-700 dark:text-gray-300">
                          {filler}%
                        </span>
                      </td>

                      {/* AI Key Strengths Snippet */}
                      <td className="py-3 px-3 max-w-xs">
                        <p className="truncate text-gray-600 dark:text-gray-400 text-[11px]">
                          {c.keyStrengths && c.keyStrengths.length > 0
                            ? c.keyStrengths[0]
                            : c.summary || "Pemahaman arsitektur komponen React yang memadai."}
                        </p>
                      </td>

                      {/* Actions */}
                      <td className="py-3 px-3 text-right">
                        <div className="flex items-center justify-end gap-1.5">
                          <button
                            onClick={() =>
                              setDrawerCandidate({
                                ...c,
                                roleLevel: c.roleLevel || "MID",
                                stage: currentStage,
                                isStarred: !!c.isStarred,
                                wpm,
                                fillerRatio: filler,
                              })
                            }
                            className="p-1.5 rounded-lg text-gray-500 hover:text-brand-600 hover:bg-gray-100 dark:hover:bg-gray-800 transition-colors"
                            title="Buka AI Executive Snapshot"
                          >
                            <Eye className="w-4 h-4" />
                          </button>

                          <button
                            onClick={() =>
                              setWhatsAppCandidate({
                                candidateId: c.candidateId,
                                roleLevel: c.roleLevel || "MID",
                                finalScore: c.finalScore,
                                keyStrengths: c.keyStrengths,
                              })
                            }
                            className="p-1.5 rounded-lg text-emerald-600 dark:text-emerald-400 hover:bg-emerald-50 dark:hover:bg-emerald-500/10 transition-colors"
                            title="Kirim Pesan WhatsApp Cepat"
                          >
                            <MessageSquare className="w-4 h-4" />
                          </button>

                          <button
                            onClick={() =>
                              setEmailCandidate({
                                candidateId: c.candidateId,
                                roleLevel: c.roleLevel || "MID",
                                finalScore: c.finalScore,
                                verbalScore: c.verbalScore,
                                keyStrengths: c.keyStrengths,
                                growthAreas: c.growthAreas,
                              })
                            }
                            className="p-1.5 rounded-lg text-gray-500 hover:text-brand-600 hover:bg-gray-100 dark:hover:bg-gray-800 transition-colors"
                            title="Kirim Email Lolos / Feedback"
                          >
                            <Mail className="w-4 h-4" />
                          </button>
                        </div>
                      </td>
                    </tr>
                  );
                })
              )}
            </tbody>
          </table>
        </div>
      </div>

      {/* Floating Bulk Action Bar */}
      <FloatingBulkBar
        selectedCount={selectedIds.size}
        onClearSelection={() => setSelectedIds(new Set())}
        onBulkStage={handleBulkStage}
        onBulkStar={handleBulkStar}
        onBulkCompare={() => setIsCompareOpen(true)}
        onBulkExport={handleBulkExport}
      />

      {/* Slide-out Candidate Drawer */}
      <CandidateDrawer
        candidate={drawerCandidate}
        isOpen={drawerCandidate !== null}
        onClose={() => setDrawerCandidate(null)}
        onUpdateStage={handleUpdateStage}
        onToggleStar={handleToggleStar}
        onSaveNotes={handleSaveNotes}
        onOpenWhatsApp={(cand) =>
          setWhatsAppCandidate({
            candidateId: cand.candidateId,
            roleLevel: cand.roleLevel,
            finalScore: cand.finalScore,
            keyStrengths: cand.keyStrengths,
          })
        }
      />

      {/* Head-to-Head Candidate Comparison Modal */}
      <CandidateCompareModal
        candidates={compareList}
        isOpen={isCompareOpen}
        onClose={() => setIsCompareOpen(false)}
        onSelectWinner={(candidateId, targetStage) => {
          handleUpdateStage(candidateId, targetStage);
          setIsCompareOpen(false);
          setSelectedIds(new Set());
        }}
      />

      {/* Automated Recruiter Email Composer Modal */}
      <RecruiterEmailModal
        candidate={emailCandidate}
        isOpen={emailCandidate !== null}
        onClose={() => setEmailCandidate(null)}
      />

      {/* 1-Click WhatsApp Quick Notification Modal */}
      <WhatsAppNotificationModal
        candidate={whatsAppCandidate}
        isOpen={whatsAppCandidate !== null}
        onClose={() => setWhatsAppCandidate(null)}
      />
    </div>
  );
}
