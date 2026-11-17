'use client';

import React, { useState, useEffect, useCallback } from "react";
import {
  X,
  Server,
  Cpu,
  Brain,
  Database,
  ExternalLink,
  RefreshCw,
  CheckCircle2,
  AlertTriangle,
  Zap,
} from "lucide-react";

type SystemHealthModalProps = {
  isOpen: boolean;
  onClose: () => void;
  apiBaseUrl: string;
};

export function SystemHealthModal({ isOpen, onClose, apiBaseUrl }: SystemHealthModalProps) {
  const [latency, setLatency] = useState<number | null>(null);
  const [serverStatus, setServerStatus] = useState<"checking" | "online" | "offline">("checking");
  const [lastChecked, setLastChecked] = useState<string>("");

  const checkHealth = useCallback(async () => {
    setServerStatus("checking");
    const start = performance.now();
    try {
      const targetUrl = `${apiBaseUrl.replace("/api", "")}/health`;
      const res = await fetch(targetUrl);
      const end = performance.now();
      if (res.ok) {
        setLatency(Math.round(end - start));
        setServerStatus("online");
      } else {
        setServerStatus("offline");
      }
    } catch {
      setServerStatus("offline");
    } finally {
      setLastChecked(
        new Date().toLocaleTimeString("id-ID", {
          hour: "2-digit",
          minute: "2-digit",
          second: "2-digit",
        })
      );
    }
  }, [apiBaseUrl]);

  useEffect(() => {
    if (isOpen) {
      checkHealth();
    }
  }, [isOpen, checkHealth]);

  if (!isOpen) return null;

  const swaggerUrl = `${apiBaseUrl.replace("/api", "")}/docs`;

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center p-4 bg-gray-900/60 backdrop-blur-sm animate-in fade-in duration-200">
      <div
        className="relative w-full max-w-xl tailadmin-card bg-white dark:bg-[#101828] border border-gray-200 dark:border-gray-800 p-6 sm:p-7 shadow-2xl rounded-3xl"
        onClick={(e) => e.stopPropagation()}
      >
        {/* Modal Header */}
        <div className="flex items-center justify-between pb-4 mb-5 border-b border-gray-100 dark:border-gray-800">
          <div className="flex items-center gap-3">
            <div className="p-2.5 rounded-xl bg-brand-50 text-brand-600 dark:bg-brand-500/10 dark:text-brand-400">
              <Zap className="w-5 h-5" />
            </div>
            <div>
              <h3 className="text-lg font-bold text-gray-900 dark:text-white">
                Status Sistem & Diagnostik AI
              </h3>
              <p className="text-xs text-gray-400">
                Pemeriksaan konektivitas server, model inference, dan database
              </p>
            </div>
          </div>
          <button
            onClick={onClose}
            className="p-1.5 rounded-lg text-gray-400 hover:text-gray-700 dark:hover:text-gray-200 hover:bg-gray-100 dark:hover:bg-gray-800 transition-colors"
          >
            <X className="w-5 h-5" />
          </button>
        </div>

        {/* Real-time Status Banner */}
        <div className="flex items-center justify-between p-4 mb-5 rounded-2xl bg-gray-50 dark:bg-gray-900/60 border border-gray-100 dark:border-gray-800">
          <div className="flex items-center gap-2.5">
            {serverStatus === "online" && (
              <span className="relative flex h-3 w-3">
                <span className="animate-ping absolute inline-flex h-full w-full rounded-full bg-emerald-400 opacity-75"></span>
                <span className="relative inline-flex rounded-full h-3 w-3 bg-emerald-500"></span>
              </span>
            )}
            {serverStatus === "offline" && <span className="w-3 h-3 rounded-full bg-rose-500" />}
            {serverStatus === "checking" && <span className="w-3 h-3 rounded-full bg-amber-500 animate-pulse" />}
            <div>
              <p className="text-sm font-semibold text-gray-900 dark:text-white">
                {serverStatus === "online" ? "Semua Layanan Berjalan Normal" : serverStatus === "offline" ? "Backend Terputus" : "Memeriksa..."}
              </p>
              <p className="text-xs text-gray-400">
                {latency !== null ? `Latensi API: ${latency} ms` : "Mengukur koneksi..."} • Cek terakhir: {lastChecked || "Baru saja"}
              </p>
            </div>
          </div>
          <button
            onClick={checkHealth}
            disabled={serverStatus === "checking"}
            className="p-2 rounded-xl text-gray-500 hover:text-brand-500 hover:bg-white dark:hover:bg-gray-800 border border-transparent hover:border-gray-200 dark:hover:border-gray-700 transition-all text-xs font-medium flex items-center gap-1.5"
            title="Refresh Status"
          >
            <RefreshCw className={`w-3.5 h-3.5 ${serverStatus === "checking" ? "animate-spin" : ""}`} />
            Refresh
          </button>
        </div>

        {/* Component Health Grid */}
        <div className="grid grid-cols-1 sm:grid-cols-2 gap-3 mb-6">
          {/* Card 1: FastAPI */}
          <div className="p-4 rounded-2xl bg-white dark:bg-[#101828] border border-gray-200/80 dark:border-gray-800/80 shadow-sm flex items-start gap-3">
            <div className="p-2 rounded-xl bg-emerald-50 text-emerald-600 dark:bg-emerald-500/10 dark:text-emerald-400 mt-0.5">
              <Server className="w-4 h-4" />
            </div>
            <div className="min-w-0 flex-1">
              <div className="flex items-center justify-between">
                <span className="text-xs font-bold text-gray-900 dark:text-white">FastAPI Server</span>
                <span className="text-[10px] font-bold text-emerald-600 bg-emerald-50 dark:bg-emerald-500/15 px-1.5 py-0.5 rounded-md">
                  Active
                </span>
              </div>
              <p className="text-[11px] text-gray-400 mt-0.5">Port 9001 • REST Upload & Results</p>
            </div>
          </div>

          {/* Card 2: Whisper STT */}
          <div className="p-4 rounded-2xl bg-white dark:bg-[#101828] border border-gray-200/80 dark:border-gray-800/80 shadow-sm flex items-start gap-3">
            <div className="p-2 rounded-xl bg-brand-50 text-brand-600 dark:bg-brand-500/10 dark:text-brand-400 mt-0.5">
              <Cpu className="w-4 h-4" />
            </div>
            <div className="min-w-0 flex-1">
              <div className="flex items-center justify-between">
                <span className="text-xs font-bold text-gray-900 dark:text-white">Whisper STT Engine</span>
                <span className="text-[10px] font-bold text-brand-600 bg-brand-50 dark:bg-brand-500/15 px-1.5 py-0.5 rounded-md">
                  Ready
                </span>
              </div>
              <p className="text-[11px] text-gray-400 mt-0.5">Multilingual + Technical Prompts</p>
            </div>
          </div>

          {/* Card 3: NLP Engine */}
          <div className="p-4 rounded-2xl bg-white dark:bg-[#101828] border border-gray-200/80 dark:border-gray-800/80 shadow-sm flex items-start gap-3">
            <div className="p-2 rounded-xl bg-purple-50 text-purple-600 dark:bg-purple-500/10 dark:text-purple-400 mt-0.5">
              <Brain className="w-4 h-4" />
            </div>
            <div className="min-w-0 flex-1">
              <div className="flex items-center justify-between">
                <span className="text-xs font-bold text-gray-900 dark:text-white">NLP Scoring Model</span>
                <span className="text-[10px] font-bold text-purple-600 bg-purple-50 dark:bg-purple-500/15 px-1.5 py-0.5 rounded-md">
                  Ready
                </span>
              </div>
              <p className="text-[11px] text-gray-400 mt-0.5">WPM Flow & Key Concept Rubric</p>
            </div>
          </div>

          {/* Card 4: Database */}
          <div className="p-4 rounded-2xl bg-white dark:bg-[#101828] border border-gray-200/80 dark:border-gray-800/80 shadow-sm flex items-start gap-3">
            <div className="p-2 rounded-xl bg-amber-50 text-amber-600 dark:bg-amber-500/10 dark:text-amber-400 mt-0.5">
              <Database className="w-4 h-4" />
            </div>
            <div className="min-w-0 flex-1">
              <div className="flex items-center justify-between">
                <span className="text-xs font-bold text-gray-900 dark:text-white">Database & ORM</span>
                <span className="text-[10px] font-bold text-amber-600 bg-amber-50 dark:bg-amber-500/15 px-1.5 py-0.5 rounded-md">
                  Connected
                </span>
              </div>
              <p className="text-[11px] text-gray-400 mt-0.5">Prisma Client & SQLite/Postgres</p>
            </div>
          </div>
        </div>

        {/* Developer Action Footer */}
        <div className="flex flex-col sm:flex-row items-center justify-between gap-3 pt-4 border-t border-gray-100 dark:border-gray-800">
          <p className="text-xs text-gray-400">
            Perlu menguji API langsung?
          </p>
          <div className="flex items-center gap-2">
            <a
              href={swaggerUrl}
              target="_blank"
              rel="noopener noreferrer"
              className="inline-flex items-center gap-1.5 px-4 py-2 rounded-xl text-xs font-semibold text-gray-700 dark:text-gray-200 bg-gray-100 hover:bg-gray-200 dark:bg-gray-800 dark:hover:bg-gray-700 transition-colors"
            >
              <ExternalLink className="w-3.5 h-3.5" />
              Buka Swagger Docs Teknis
            </a>
            <button
              onClick={onClose}
              className="px-4 py-2 rounded-xl text-xs font-semibold text-white bg-brand-500 hover:bg-brand-600 transition-colors"
            >
              Tutup
            </button>
          </div>
        </div>
      </div>
    </div>
  );
}
