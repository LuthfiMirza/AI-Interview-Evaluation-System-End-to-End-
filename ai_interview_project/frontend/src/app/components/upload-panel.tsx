'use client';

import { FormEvent, useMemo, useState } from "react";
import { UploadCloud, FileVideo, CheckCircle2, AlertCircle, Sparkles, ChevronDown, RefreshCw } from "lucide-react";

type UploadPanelProps = {
  apiBaseUrl: string;
};

type UploadStatus = "idle" | "uploading" | "success" | "error";

const PRESET_ROLES = {
  "Frontend Engineer (React.js & Next.js)": {
    junior: {
      label: "Junior Level",
      expected: "Menguasai React.js, Next.js, dan TypeScript, serta memahami arsitektur Component, Routing (App Router/Pages Router), State Management (Zustand/Redux/Context API), dan Data Fetching/API Integration.",
      concepts: ["Component Architecture", "Routing (App/Pages)", "State Management", "Data Fetching / REST", "TypeScript"],
    },
    mid: {
      label: "Mid Level",
      expected: "Menguasai React.js dan Next.js tingkat lanjut dengan TypeScript, arsitektur Componentisasi skalabel, optimasi performa render (memo, lazy loading, SSR/SSG/ISR), global State Management (Zustand/RTK), routing dinamis, dan strategi data fetching yang handal.",
      concepts: ["Component Architecture", "SSR / SSG / ISR", "Performance Optimization", "State Management (Zustand/RTK)", "Testing & Accessibility"],
    },
    senior: {
      label: "Senior Level",
      expected: "Kepemimpinan arsitektur frontend skala besar dengan React dan Next.js: Design System, Micro-frontend atau monorepo, optimasi Core Web Vitals, observability & error tracking, strategi testing komprehensif, dan arsitektur state terdistribusi.",
      concepts: ["Design System", "Core Web Vitals", "Distributed State", "E2E Testing Strategy", "Observability / Monitoring"],
    },
  },
};

export function UploadPanel({ apiBaseUrl }: UploadPanelProps) {
  const [level, setLevel] = useState<"junior" | "mid" | "senior">("junior");
  const [candidateId, setCandidateId] = useState("");
  const [expectedAnswer, setExpectedAnswer] = useState(
    PRESET_ROLES["Frontend Engineer (React.js & Next.js)"]["junior"].expected
  );
  const [isCustomPrompt, setIsCustomPrompt] = useState(false);
  const [file, setFile] = useState<File | null>(null);
  const [isDragging, setIsDragging] = useState(false);
  const [status, setStatus] = useState<UploadStatus>("idle");
  const [message, setMessage] = useState<string | null>(null);
  const [interviewId, setInterviewId] = useState<string | null>(null);

  const disabled = useMemo(() => status === "uploading", [status]);

  const handleLevelChange = (newLevel: "junior" | "mid" | "senior") => {
    setLevel(newLevel);
    if (!isCustomPrompt) {
      setExpectedAnswer(PRESET_ROLES["Frontend Engineer (React.js & Next.js)"][newLevel].expected);
    }
  };

  const handleDrop = (e: React.DragEvent) => {
    e.preventDefault();
    setIsDragging(false);
    if (e.dataTransfer.files && e.dataTransfer.files[0]) {
      const droppedFile = e.dataTransfer.files[0];
      if (droppedFile.type.startsWith("video/")) {
        setFile(droppedFile);
      } else {
        setStatus("error");
        setMessage("Format file harus berupa video (.mp4, .webm, .mov)");
      }
    }
  };

  async function handleSubmit(event: FormEvent<HTMLFormElement>) {
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

  const currentConcepts = PRESET_ROLES["Frontend Engineer (React.js & Next.js)"][level].concepts;

  return (
    <div className="space-y-6">
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

      <form className="space-y-5" onSubmit={handleSubmit}>
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
            onDrop={handleDrop}
            className={`relative flex flex-col items-center justify-center p-8 rounded-2xl border-2 border-dashed transition-all cursor-pointer ${
              isDragging
                ? "border-brand-500 bg-brand-50/50 dark:bg-brand-500/5"
                : "border-gray-300 dark:border-gray-700 bg-gray-50/50 dark:bg-gray-900/30 hover:border-brand-400"
            }`}
            onClick={() => document.getElementById("file-upload-input")?.click()}
          >
            <input
              id="file-upload-input"
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
                  <p className="text-sm font-semibold text-gray-900 dark:text-white">
                    {file.name}
                  </p>
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
                <p className="text-xs text-gray-400">
                  Mendukung format MP4, WebM, MOV (Maksimal 1 GB)
                </p>
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
            <div>
              <p className="font-medium">{message}</p>
              {interviewId && (
                <p className="mt-1 text-xs opacity-90">
                  ID: <span className="font-mono font-bold">{interviewId}</span>
                </p>
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
    </div>
  );
}
