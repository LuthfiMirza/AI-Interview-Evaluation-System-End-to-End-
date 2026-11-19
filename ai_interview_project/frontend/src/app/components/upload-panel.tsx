'use client';

import { FormEvent, useMemo, useState } from "react";
import Link from "next/link";
import {
  UploadCloud,
  FileVideo,
  CheckCircle2,
  AlertCircle,
  Sparkles,
  RefreshCw,
  Layers,
  ArrowRight,
  Trash2,
  ListOrdered,
  UserCheck,
} from "lucide-react";

type UploadPanelProps = {
  apiBaseUrl: string;
};

type UploadStatus = "idle" | "uploading" | "success" | "error";

type BulkQueueItem = {
  id: string;
  file: File;
  candidateId: string;
  status: "pending" | "uploading" | "completed" | "failed";
  interviewId?: string;
  error?: string;
};

const PRESET_ROLES = {
  "Frontend Engineer (React.js & Next.js)": {
    junior: {
      label: "Junior Level",
      expected:
        "Menguasai React.js, Next.js, dan TypeScript, serta memahami arsitektur Component, Routing (App Router/Pages Router), State Management (Zustand/Redux/Context API), dan Data Fetching/API Integration.",
      concepts: ["Component Architecture", "Routing (App/Pages)", "State Management", "Data Fetching / REST", "TypeScript"],
    },
    mid: {
      label: "Mid Level",
      expected:
        "Menguasai React.js dan Next.js tingkat lanjut dengan TypeScript, arsitektur Componentisasi skalabel, optimasi performa render (memo, lazy loading, SSR/SSG/ISR), global State Management (Zustand/RTK), routing dinamis, dan strategi data fetching yang handal.",
      concepts: ["Component Architecture", "SSR / SSG / ISR", "Performance Optimization", "State Management (Zustand/RTK)", "Testing & Accessibility"],
    },
    senior: {
      label: "Senior Level",
      expected:
        "Kepemimpinan arsitektur frontend skala besar dengan React dan Next.js: Design System, Micro-frontend atau monorepo, optimasi Core Web Vitals, observability & error tracking, strategi testing komprehensif, dan arsitektur state terdistribusi.",
      concepts: ["Design System", "Core Web Vitals", "Distributed State", "E2E Testing Strategy", "Observability / Monitoring"],
    },
  },
};

export function UploadPanel({ apiBaseUrl }: UploadPanelProps) {
  // Mode switcher: "single" vs "bulk"
  const [uploadMode, setUploadMode] = useState<"single" | "bulk">("single");

  // Common Level selection
  const [level, setLevel] = useState<"junior" | "mid" | "senior">("mid");
  const [expectedAnswer, setExpectedAnswer] = useState(
    PRESET_ROLES["Frontend Engineer (React.js & Next.js)"]["mid"].expected
  );
  const [isCustomPrompt, setIsCustomPrompt] = useState(false);

  // Single mode state
  const [candidateId, setCandidateId] = useState("");
  const [file, setFile] = useState<File | null>(null);
  const [isDragging, setIsDragging] = useState(false);
  const [status, setStatus] = useState<UploadStatus>("idle");
  const [message, setMessage] = useState<string | null>(null);
  const [interviewId, setInterviewId] = useState<string | null>(null);

  // Bulk mode queue state
  const [bulkQueue, setBulkQueue] = useState<BulkQueueItem[]>([]);
  const [isBulkProcessing, setIsBulkProcessing] = useState(false);
  const [bulkCompletedCount, setBulkCompletedCount] = useState(0);

  const disabled = useMemo(
    () => status === "uploading" || isBulkProcessing,
    [status, isBulkProcessing]
  );

  const handleLevelChange = (newLevel: "junior" | "mid" | "senior") => {
    setLevel(newLevel);
    if (!isCustomPrompt) {
      setExpectedAnswer(PRESET_ROLES["Frontend Engineer (React.js & Next.js)"][newLevel].expected);
    }
  };

  // Single file drop handler
  const handleSingleDrop = (e: React.DragEvent) => {
    e.preventDefault();
    setIsDragging(false);
    if (e.dataTransfer.files && e.dataTransfer.files[0]) {
      const droppedFile = e.dataTransfer.files[0];
      if (droppedFile.type.startsWith("video/") || droppedFile.name.match(/\.(mp4|webm|mov|mkv)$/i)) {
        setFile(droppedFile);
      } else {
        setStatus("error");
        setMessage("Format file harus berupa video (.mp4, .webm, .mov, .mkv)");
      }
    }
  };

  // Bulk multiple files drop handler
  const handleBulkDrop = (e: React.DragEvent) => {
    e.preventDefault();
    setIsDragging(false);
    if (e.dataTransfer.files && e.dataTransfer.files.length > 0) {
      addFilesToBulkQueue(Array.from(e.dataTransfer.files));
    }
  };

  const addFilesToBulkQueue = (newFiles: File[]) => {
    const videoFiles = newFiles.filter(
      (f) => f.type.startsWith("video/") || f.name.match(/\.(mp4|webm|mov|mkv)$/i)
    );

    const newItems: BulkQueueItem[] = videoFiles.map((f, idx) => {
      // derive candidate ID from filename or counter
      const baseName = f.name.replace(/\.[^/.]+$/, "").replace(/[^a-zA-Z0-9_-]/g, "-");
      const candId = `CAND-${baseName.slice(0, 16).toUpperCase() || (bulkQueue.length + idx + 1).toString().padStart(3, "0")}`;
      return {
        id: Math.random().toString(36).substring(2, 9),
        file: f,
        candidateId: candId,
        status: "pending",
      };
    });

    setBulkQueue((prev) => [...prev, ...newItems]);
  };

  const removeBulkItem = (id: string) => {
    setBulkQueue((prev) => prev.filter((item) => item.id !== id));
  };

  // Single Upload Form Submission
  async function handleSingleSubmit(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();
    if (!file) {
      setStatus("error");
      setMessage("Silakan lampirkan file video wawancara terlebih dahulu.");
      return;
    }

    try {
      setStatus("uploading");
      setMessage("Mengunggah video dan memproses evaluasi background (Whisper STT + NLP)...");

      const formData = new FormData();
      formData.append("file", file);
      formData.append("role_level", level.toUpperCase());
      if (candidateId.trim()) {
        formData.append("candidate_id", candidateId.trim());
      }
      if (expectedAnswer.trim()) {
        formData.append("expected_answer", expectedAnswer.trim());
      }

      const response = await fetch(`${apiBaseUrl}/interviews/upload`, {
        method: "POST",
        body: formData,
      });

      if (!response.ok) {
        throw new Error(`Upload gagal dengan status HTTP ${response.status}`);
      }

      const data = (await response.json()) as { interview_id: string };
      setInterviewId(data.interview_id);
      setStatus("success");
      setMessage(
        `Video berhasil diunggah! ID Sesi: ${data.interview_id}. Pipeline AI sedang mengevaluasi transkrip dan kelancaran.`
      );
      setFile(null);
    } catch (error) {
      console.error(error);
      setStatus("error");
      setMessage(
        error instanceof Error ? error.message : "Terjadi kesalahan saat mengunggah video."
      );
    }
  }

  // Bulk Process Execution Queue
  async function handleProcessBulkQueue() {
    if (bulkQueue.length === 0 || isBulkProcessing) return;

    setIsBulkProcessing(true);
    let completed = 0;

    for (let i = 0; i < bulkQueue.length; i++) {
      const item = bulkQueue[i];
      if (item.status === "completed") {
        completed++;
        continue;
      }

      // Mark current as uploading
      setBulkQueue((prev) =>
        prev.map((q, idx) => (idx === i ? { ...q, status: "uploading" } : q))
      );

      try {
        const formData = new FormData();
        formData.append("file", item.file);
        formData.append("candidate_id", item.candidateId);
        formData.append("role_level", level.toUpperCase());
        formData.append("expected_answer", expectedAnswer);

        const response = await fetch(`${apiBaseUrl}/interviews/upload`, {
          method: "POST",
          body: formData,
        });

        if (!response.ok) {
          throw new Error(`HTTP ${response.status}`);
        }

        const data = (await response.json()) as { interview_id: string };
        completed++;
        setBulkCompletedCount(completed);

        setBulkQueue((prev) =>
          prev.map((q, idx) =>
            idx === i ? { ...q, status: "completed", interviewId: data.interview_id } : q
          )
        );
      } catch (err: any) {
        setBulkQueue((prev) =>
          prev.map((q, idx) =>
            idx === i ? { ...q, status: "failed", error: err.message || "Gagal upload" } : q
          )
        );
      }
    }

    setIsBulkProcessing(false);
  }

  const currentConcepts = PRESET_ROLES["Frontend Engineer (React.js & Next.js)"][level].concepts;
  const bulkOverallPct =
    bulkQueue.length > 0 ? Math.round((bulkCompletedCount / bulkQueue.length) * 100) : 0;

  return (
    <div className="space-y-6">
      {/* Upload Mode Switcher Tabs */}
      <div className="flex items-center justify-between p-1.5 bg-gray-100 dark:bg-gray-800/80 rounded-2xl text-xs font-semibold">
        <button
          type="button"
          onClick={() => setUploadMode("single")}
          className={`flex-1 py-2 px-4 rounded-xl transition-all flex items-center justify-center gap-2 ${
            uploadMode === "single"
              ? "bg-white dark:bg-gray-700 text-brand-600 dark:text-brand-400 shadow-sm"
              : "text-gray-500 hover:text-gray-900 dark:text-gray-400"
          }`}
        >
          <FileVideo className="w-4 h-4" />
          Satu Video (Single Mode)
        </button>

        <button
          type="button"
          onClick={() => setUploadMode("bulk")}
          className={`flex-1 py-2 px-4 rounded-xl transition-all flex items-center justify-center gap-2 ${
            uploadMode === "bulk"
              ? "bg-white dark:bg-gray-700 text-purple-600 dark:text-purple-400 shadow-sm"
              : "text-gray-500 hover:text-gray-900 dark:text-gray-400"
          }`}
        >
          <Layers className="w-4 h-4" />
          Unggah Massal (Bulk Queue Mode)
          <span className="px-1.5 py-0.2 rounded-full text-[10px] font-bold bg-purple-100 text-purple-700 dark:bg-purple-500/20 dark:text-purple-300">
            Batch
          </span>
        </button>
      </div>

      {/* Role & Level Preset Selector */}
      <div className="p-4 rounded-xl bg-gray-50 dark:bg-gray-900/60 border border-gray-100 dark:border-gray-800">
        <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-3 mb-3">
          <div className="flex items-center gap-2">
            <Sparkles className="w-4 h-4 text-brand-500" />
            <span className="text-sm font-semibold text-gray-900 dark:text-white">
              Target Kualifikasi: Frontend Engineer (React.js & Next.js)
            </span>
          </div>
          {/* Level Tabs */}
          <div className="flex items-center p-1 rounded-lg bg-gray-200/70 dark:bg-gray-800 self-start sm:self-auto">
            {(["junior", "mid", "senior"] as const).map((lvl) => (
              <button
                key={lvl}
                type="button"
                onClick={() => handleLevelChange(lvl)}
                className={`px-3 py-1 rounded-md text-xs font-semibold capitalize transition-all ${
                  level === lvl
                    ? "bg-white dark:bg-brand-500 text-brand-600 dark:text-white shadow-sm"
                    : "text-gray-500 hover:text-gray-900 dark:text-gray-400 dark:hover:text-white"
                }`}
              >
                {lvl}
              </button>
            ))}
          </div>
        </div>

        {/* Expected Concepts Badges */}
        <div className="flex flex-wrap gap-1.5 items-center">
          <span className="text-xs text-gray-400 mr-1">Rubrik Poin Kunci:</span>
          {currentConcepts.map((c) => (
            <span
              key={c}
              className="px-2.5 py-0.5 rounded-full text-xs font-medium bg-white dark:bg-gray-800 text-gray-700 dark:text-gray-300 border border-gray-200 dark:border-gray-700"
            >
              {c}
            </span>
          ))}
        </div>
      </div>

      {/* MODE 1: SINGLE CANDIDATE MODE */}
      {uploadMode === "single" ? (
        <form className="space-y-5" onSubmit={handleSingleSubmit}>
          <div className="grid grid-cols-1 sm:grid-cols-2 gap-4">
            <div>
              <label className="block text-xs font-semibold text-gray-700 dark:text-gray-300 mb-1.5">
                Candidate Identifier
              </label>
              <input
                type="text"
                placeholder="e.g. CAND-FRONTEND-001"
                value={candidateId}
                onChange={(e) => setCandidateId(e.target.value)}
                disabled={disabled}
                className="w-full h-11 px-3.5 rounded-xl text-sm bg-gray-50 dark:bg-gray-900 border border-gray-200 dark:border-gray-800 text-gray-900 dark:text-white focus:outline-none focus:ring-2 focus:ring-brand-500/20 focus:border-brand-500"
              />
            </div>

            <div>
              <div className="flex items-center justify-between mb-1.5">
                <label className="text-xs font-semibold text-gray-700 dark:text-gray-300">
                  Mode Ekspektasi Jawaban
                </label>
                <button
                  type="button"
                  onClick={() => setIsCustomPrompt(!isCustomPrompt)}
                  className="text-xs font-medium text-brand-600 dark:text-brand-400 hover:underline"
                >
                  {isCustomPrompt ? "Gunakan Preset" : "Edit Kustom"}
                </button>
              </div>
              <input
                type="text"
                readOnly={!isCustomPrompt}
                value={expectedAnswer}
                onChange={(e) => setExpectedAnswer(e.target.value)}
                disabled={disabled}
                className={`w-full h-11 px-3.5 rounded-xl text-sm border text-gray-900 dark:text-white focus:outline-none transition-all ${
                  isCustomPrompt
                    ? "bg-white dark:bg-gray-900 border-brand-500 focus:ring-2 focus:ring-brand-500/20"
                    : "bg-gray-100/70 dark:bg-gray-800/60 border-gray-200 dark:border-gray-800 text-gray-500"
                }`}
              />
            </div>
          </div>

          {/* Drag and Drop Zone */}
          <div>
            <label className="block text-xs font-semibold text-gray-700 dark:text-gray-300 mb-1.5">
              File Rekaman Video Wawancara
            </label>
            <div
              onDragOver={(e) => {
                e.preventDefault();
                setIsDragging(true);
              }}
              onDragLeave={() => setIsDragging(false)}
              onDrop={handleSingleDrop}
              className={`relative flex flex-col items-center justify-center p-8 rounded-2xl border-2 border-dashed transition-all cursor-pointer ${
                isDragging
                  ? "border-brand-500 bg-brand-50/50 dark:bg-brand-500/5"
                  : "border-gray-300 dark:border-gray-700 bg-gray-50/50 dark:bg-gray-900/30 hover:border-brand-400"
              }`}
              onClick={() => document.getElementById("single-upload-input")?.click()}
            >
              <input
                id="single-upload-input"
                type="file"
                accept="video/mp4,video/webm,video/quicktime,video/mkv"
                className="hidden"
                onChange={(e) => setFile(e.target.files?.[0] ?? null)}
                disabled={disabled}
              />

              {file ? (
                <div className="flex items-center gap-3">
                  <div className="p-3 rounded-xl bg-brand-50 text-brand-600 dark:bg-brand-500/10 dark:text-brand-400">
                    <FileVideo className="w-8 h-8" />
                  </div>
                  <div>
                    <p className="text-sm font-semibold text-gray-900 dark:text-white">{file.name}</p>
                    <p className="text-xs text-gray-400">
                      {(file.size / (1024 * 1024)).toFixed(2)} MB • Siap diproses
                    </p>
                  </div>
                </div>
              ) : (
                <div className="flex flex-col items-center text-center">
                  <div className="p-3 mb-3 rounded-xl bg-gray-100 dark:bg-gray-800 text-gray-500 dark:text-gray-400">
                    <UploadCloud className="w-6 h-6" />
                  </div>
                  <p className="text-sm font-semibold text-gray-900 dark:text-white mb-1">
                    Drag and drop file rekaman di sini, atau <span className="text-brand-500">Pilih File</span>
                  </p>
                  <p className="text-xs text-gray-400">Mendukung format MP4, WebM, MOV, MKV (Maksimal 1 GB)</p>
                </div>
              )}
            </div>
          </div>

          {/* Feedback Messages */}
          {message && (
            <div
              className={`p-4 rounded-xl flex items-start gap-3 text-sm ${
                status === "success"
                  ? "bg-emerald-50 text-emerald-800 dark:bg-emerald-950/40 dark:text-emerald-300 border border-emerald-200 dark:border-emerald-800"
                  : status === "error"
                  ? "bg-rose-50 text-rose-800 dark:bg-rose-950/40 dark:text-rose-300 border border-rose-200 dark:border-rose-800"
                  : "bg-blue-50 text-blue-800 dark:bg-blue-950/40 dark:text-blue-300 border border-blue-200 dark:border-blue-800"
              }`}
            >
              {status === "success" && <CheckCircle2 className="w-5 h-5 shrink-0 text-emerald-600 mt-0.5" />}
              {status === "error" && <AlertCircle className="w-5 h-5 shrink-0 text-rose-600 mt-0.5" />}
              {status === "uploading" && <RefreshCw className="w-5 h-5 shrink-0 text-blue-600 animate-spin mt-0.5" />}
              <div className="flex-1">
                <p className="font-medium">{message}</p>
                {interviewId && (
                  <div className="mt-2 flex items-center gap-3">
                    <span className="text-xs opacity-90 font-mono font-bold">ID: {interviewId}</span>
                    <Link
                      href="/candidates"
                      className="text-xs font-semibold text-brand-600 dark:text-brand-400 hover:underline flex items-center gap-1"
                    >
                      Buka di Direktori Kandidat <ArrowRight className="w-3.5 h-3.5" />
                    </Link>
                  </div>
                )}
              </div>
            </div>
          )}

          {/* Action Button */}
          <div className="flex justify-end">
            <button
              type="submit"
              disabled={disabled || !file}
              className="flex items-center gap-2 px-6 py-2.5 rounded-xl font-medium text-sm text-white bg-brand-500 hover:bg-brand-600 active:bg-brand-700 disabled:opacity-50 disabled:cursor-not-allowed shadow-md shadow-brand-500/20 transition-all"
            >
              {status === "uploading" ? (
                <>
                  <RefreshCw className="w-4 h-4 animate-spin" />
                  Mengevaluasi...
                </>
              ) : (
                <>
                  <Sparkles className="w-4 h-4" />
                  Mulai Evaluasi AI
                </>
              )}
            </button>
          </div>
        </form>
      ) : (
        /* MODE 2: BULK QUEUE MODE */
        <div className="space-y-5">
          {/* Multi-file Drag and Drop Zone */}
          <div
            onDragOver={(e) => {
              e.preventDefault();
              setIsDragging(true);
            }}
            onDragLeave={() => setIsDragging(false)}
            onDrop={handleBulkDrop}
            onClick={() => document.getElementById("bulk-upload-input")?.click()}
            className={`relative flex flex-col items-center justify-center p-8 rounded-2xl border-2 border-dashed transition-all cursor-pointer ${
              isDragging
                ? "border-purple-500 bg-purple-50/50 dark:bg-purple-500/5"
                : "border-gray-300 dark:border-gray-700 bg-gray-50/50 dark:bg-gray-900/30 hover:border-purple-400"
            }`}
          >
            <input
              id="bulk-upload-input"
              type="file"
              multiple
              accept="video/mp4,video/webm,video/quicktime,video/mkv"
              className="hidden"
              onChange={(e) => {
                if (e.target.files) addFilesToBulkQueue(Array.from(e.target.files));
              }}
              disabled={isBulkProcessing}
            />

            <div className="flex flex-col items-center text-center">
              <div className="p-3 mb-3 rounded-xl bg-purple-50 text-purple-600 dark:bg-purple-500/10 dark:text-purple-400">
                <UploadCloud className="w-6 h-6" />
              </div>
              <p className="text-sm font-semibold text-gray-900 dark:text-white mb-1">
                Drag and drop <span className="text-purple-600 dark:text-purple-400">banyak video</span> sekaligus di sini
              </p>
              <p className="text-xs text-gray-400">
                Mendukung seleksi 2–20 video sekaligus (.mp4, .webm, .mov, .mkv)
              </p>
            </div>
          </div>

          {/* Queue List Table */}
          {bulkQueue.length > 0 && (
            <div className="space-y-3">
              <div className="flex items-center justify-between">
                <div className="flex items-center gap-2">
                  <ListOrdered className="w-4 h-4 text-purple-600" />
                  <span className="text-xs font-bold text-gray-900 dark:text-white uppercase tracking-wider">
                    Daftar Antrean Video ({bulkQueue.length} Berkas)
                  </span>
                </div>
                {bulkCompletedCount > 0 && (
                  <span className="text-xs font-semibold text-emerald-600 dark:text-emerald-400">
                    {bulkCompletedCount} dari {bulkQueue.length} Selesai ({bulkOverallPct}%)
                  </span>
                )}
              </div>

              {/* Progress Bar */}
              {isBulkProcessing && (
                <div className="w-full h-2 rounded-full bg-gray-100 dark:bg-gray-800 overflow-hidden">
                  <div
                    className="h-full bg-gradient-to-r from-purple-500 to-brand-500 transition-all duration-300"
                    style={{ width: `${bulkOverallPct}%` }}
                  />
                </div>
              )}

              {/* Queue Items Container */}
              <div className="border border-gray-200 dark:border-gray-800 rounded-xl overflow-hidden bg-white dark:bg-gray-900/40 divide-y divide-gray-100 dark:divide-gray-800 text-xs">
                {bulkQueue.map((item, idx) => (
                  <div key={item.id} className="p-3 flex items-center justify-between gap-3">
                    <div className="flex items-center gap-3 min-w-0">
                      <span className="w-5 h-5 rounded-full bg-gray-100 dark:bg-gray-800 font-bold text-[10px] text-gray-500 flex items-center justify-center shrink-0">
                        {idx + 1}
                      </span>
                      <div className="min-w-0">
                        <p className="font-semibold text-gray-900 dark:text-white truncate">
                          {item.file.name}
                        </p>
                        <p className="text-[11px] text-gray-400">
                          {(item.file.size / (1024 * 1024)).toFixed(2)} MB • ID:{" "}
                          <span className="font-mono text-brand-600 dark:text-brand-400 font-bold">
                            {item.candidateId}
                          </span>
                        </p>
                      </div>
                    </div>

                    <div className="flex items-center gap-2 shrink-0">
                      {item.status === "pending" && (
                        <span className="px-2 py-0.5 rounded-md font-semibold text-[10px] bg-gray-100 text-gray-600 dark:bg-gray-800 dark:text-gray-400">
                          Menunggu
                        </span>
                      )}
                      {item.status === "uploading" && (
                        <span className="px-2 py-0.5 rounded-md font-semibold text-[10px] bg-blue-50 text-blue-600 dark:bg-blue-500/15 dark:text-blue-400 flex items-center gap-1">
                          <RefreshCw className="w-3 h-3 animate-spin" />
                          Memproses AI
                        </span>
                      )}
                      {item.status === "completed" && (
                        <span className="px-2 py-0.5 rounded-md font-semibold text-[10px] bg-emerald-50 text-emerald-600 dark:bg-emerald-500/15 dark:text-emerald-400 flex items-center gap-1">
                          <CheckCircle2 className="w-3 h-3" />
                          Selesai
                        </span>
                      )}
                      {item.status === "failed" && (
                        <span className="px-2 py-0.5 rounded-md font-semibold text-[10px] bg-rose-50 text-rose-600 dark:bg-rose-500/15 dark:text-rose-400">
                          Gagal
                        </span>
                      )}

                      {!isBulkProcessing && item.status !== "completed" && (
                        <button
                          type="button"
                          onClick={() => removeBulkItem(item.id)}
                          className="p-1 rounded text-gray-400 hover:text-rose-500"
                        >
                          <Trash2 className="w-3.5 h-3.5" />
                        </button>
                      )}
                    </div>
                  </div>
                ))}
              </div>

              {/* Action: Trigger Batch Upload */}
              <div className="flex items-center justify-between pt-2">
                <button
                  type="button"
                  onClick={() => setBulkQueue([])}
                  disabled={isBulkProcessing}
                  className="text-xs text-gray-400 hover:text-gray-600 dark:hover:text-gray-200"
                >
                  Kosongkan Antrean
                </button>

                <div className="flex items-center gap-2">
                  {bulkCompletedCount > 0 && (
                    <Link
                      href="/candidates"
                      className="inline-flex items-center gap-1.5 px-4 py-2 rounded-xl text-xs font-semibold text-brand-600 dark:text-brand-400 bg-brand-50 dark:bg-brand-500/10 hover:bg-brand-100 transition-colors"
                    >
                      Lihat Hasil di Direktori <ArrowRight className="w-3.5 h-3.5" />
                    </Link>
                  )}

                  <button
                    type="button"
                    onClick={handleProcessBulkQueue}
                    disabled={isBulkProcessing || bulkQueue.length === 0}
                    className="inline-flex items-center gap-2 px-5 py-2.5 rounded-xl font-bold text-xs text-white bg-purple-600 hover:bg-purple-500 disabled:opacity-50 disabled:cursor-not-allowed shadow-md shadow-purple-500/20 transition-all"
                  >
                    {isBulkProcessing ? (
                      <>
                        <RefreshCw className="w-4 h-4 animate-spin" />
                        Memproses Batch... ({bulkCompletedCount}/{bulkQueue.length})
                      </>
                    ) : (
                      <>
                        <Sparkles className="w-4 h-4" />
                        Mulai Proses Antrean Batch ({bulkQueue.length} Video)
                      </>
                    )}
                  </button>
                </div>
              </div>
            </div>
          )}
        </div>
      )}
    </div>
  );
}
