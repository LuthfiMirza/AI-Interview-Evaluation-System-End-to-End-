import { DashboardShell } from "../components/dashboard-shell";
import { Code2, CheckCircle2, Sparkles, BookOpen, Layers, Target } from "lucide-react";

export const dynamic = "force-static";

export default function RubricsPage() {
  const apiBase = process.env.NEXT_PUBLIC_API_BASE_URL ?? "http://localhost:8000/api";

  const rubrics = [
    {
      level: "Junior Frontend Engineer",
      threshold: "75%",
      color: "border-brand-200 dark:border-brand-900/50 bg-brand-50/20",
      badgeColor: "bg-brand-50 text-brand-600 dark:bg-brand-500/15 dark:text-brand-400",
      description:
        "Menguasai React.js, Next.js, dan TypeScript, serta memahami arsitektur Component, Routing (App Router/Pages Router), State Management (Zustand/Redux/Context API), dan Data Fetching/API Integration.",
      concepts: [
        "Component Architecture (Props & Reusability)",
        "Next.js Routing (App Router / Pages Router)",
        "State Management (Zustand / Context API / Redux)",
        "Data Fetching (REST API / Fetch / useEffect)",
        "TypeScript Basics (Types & Interfaces)",
      ],
      focus: "Kemampuan membangun antarmuka web modern dengan React/Next.js, pemisahan komponen rapi, navigasi rute, dan konsumsi REST API dasar.",
    },
    {
      level: "Mid Frontend Engineer",
      threshold: "80%",
      color: "border-emerald-200 dark:border-emerald-900/50 bg-emerald-50/20",
      badgeColor: "bg-emerald-50 text-emerald-600 dark:bg-emerald-500/15 dark:text-emerald-400",
      description:
        "Menguasai React.js dan Next.js tingkat lanjut dengan TypeScript, arsitektur Componentisasi skalabel, optimasi performa render (memo, lazy loading, SSR/SSG/ISR), global State Management (Zustand/RTK), routing dinamis, dan strategi data fetching yang handal.",
      concepts: [
        "Advanced Component Pattern (Compound Components / Custom Hooks)",
        "Rendering Strategies (SSR, SSG, ISR, Hydration)",
        "Performance Optimization (React.memo, useMemo, Dynamic Imports)",
        "Global State Management Architecture (Zustand / Redux Toolkit)",
        "Testing & Web Accessibility (RTL, Jest, Semantic HTML)",
      ],
      focus: "Pemahaman mendalam tentang siklus render, optimasi performa frontend, strategi rendering server vs client, serta arsitektur state yang rapi.",
    },
    {
      level: "Senior Frontend Engineer",
      threshold: "85%",
      color: "border-purple-200 dark:border-purple-900/50 bg-purple-50/20",
      badgeColor: "bg-purple-50 text-purple-600 dark:bg-purple-500/15 dark:text-purple-400",
      description:
        "Kepemimpinan arsitektur frontend skala besar dengan React dan Next.js: Design System, Micro-frontend atau monorepo, optimasi Core Web Vitals, observability & error tracking, strategi testing komprehensif, dan arsitektur state terdistribusi.",
      concepts: [
        "Design System & Component Library Architecture",
        "Performance Profiling & Core Web Vitals (LCP, FID, CLS)",
        "Complex State & Caching (TanStack Query, SWR, Server Actions)",
        "End-to-End Testing & CI/CD Delivery Pipeline",
        "Frontend Observability, Error Tracking & Logging",
      ],
      focus: "Skalabilitas frontend skala enterprise, arsitektur design system, kepemimpinan teknis dalam optimasi web vitals, dan standar delivery.",
    },
  ];

  return (
    <DashboardShell apiBaseUrl={apiBase}>
      {/* Header */}
      <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-4 mb-8">
        <div>
          <div className="flex items-center gap-2 mb-1">
            <h1 className="text-2xl sm:text-3xl font-extrabold tracking-tight text-gray-900 dark:text-white">
              Standar Rubrik & Kriteria Evaluasi
            </h1>
            <span className="px-2.5 py-0.5 rounded-full text-xs font-semibold bg-brand-50 text-brand-600 dark:bg-brand-500/15 dark:text-brand-400">
              React & Next.js
            </span>
          </div>
          <p className="text-sm text-gray-500 dark:text-gray-400">
            Panduan kualifikasi terstruktur dan daftar poin kunci (*key concepts*) yang dijadikan tolok ukur oleh sistem AI.
          </p>
        </div>
      </div>

      {/* 3 Level Rubric Cards */}
      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6 mb-8">
        {rubrics.map((r, idx) => (
          <div
            key={idx}
            className={`tailadmin-card p-6 sm:p-7 flex flex-col justify-between border-2 ${r.color}`}
          >
            <div>
              {/* Header Card */}
              <div className="flex items-center justify-between mb-4">
                <span className={`px-3 py-1 rounded-full text-xs font-bold ${r.badgeColor}`}>
                  {r.level}
                </span>
                <span className="text-xs font-semibold text-gray-500 dark:text-gray-400">
                  Passing: <strong className="text-gray-900 dark:text-white">{r.threshold}</strong>
                </span>
              </div>

              {/* Target Description */}
              <div className="mb-5">
                <span className="text-[11px] font-bold uppercase tracking-wider text-gray-400 block mb-1">
                  Target Ekspektasi Jawaban
                </span>
                <p className="text-xs text-gray-700 dark:text-gray-300 leading-relaxed italic bg-white/70 dark:bg-gray-900/50 p-3 rounded-xl border border-gray-100 dark:border-gray-800">
                  &ldquo;{r.description}&rdquo;
                </p>
              </div>

              {/* Key Concepts Checklist */}
              <div className="mb-5">
                <span className="text-[11px] font-bold uppercase tracking-wider text-gray-400 block mb-2 flex items-center gap-1.5">
                  <CheckCircle2 className="w-3.5 h-3.5 text-brand-500" />
                  Poin Kunci Checklist (*Rubric Concepts*)
                </span>
                <ul className="space-y-1.5 text-xs text-gray-600 dark:text-gray-300">
                  {r.concepts.map((c, i) => (
                    <li key={i} className="flex items-start gap-2">
                      <span className="text-brand-500 font-bold">•</span>
                      <span>{c}</span>
                    </li>
                  ))}
                </ul>
              </div>
            </div>

            {/* Focus Summary Footer */}
            <div className="pt-4 border-t border-gray-100 dark:border-gray-800/80 text-[11px] text-gray-500 dark:text-gray-400">
              <strong>Fokus HR:</strong> {r.focus}
            </div>
          </div>
        ))}
      </div>

      {/* Whisper Technical Vocabulary Injection Box */}
      <div className="tailadmin-card p-6 sm:p-7">
        <div className="flex items-center gap-3 mb-4">
          <div className="p-2.5 rounded-xl bg-purple-50 text-purple-600 dark:bg-purple-500/10 dark:text-purple-400">
            <BookOpen className="w-5 h-5" />
          </div>
          <div>
            <h3 className="text-base font-bold text-gray-900 dark:text-white">
              Kamus Jargon Teknis Whisper STT (*initial_prompt*)
            </h3>
            <p className="text-xs text-gray-400">
              Daftar kosakata teknologi yang diinjeksikan secara otomatis ke model Whisper agar transkrip tidak salah eja:
            </p>
          </div>
        </div>

        <div className="flex flex-wrap gap-2 p-4 rounded-2xl bg-gray-50 dark:bg-gray-900/60 border border-gray-100 dark:border-gray-800">
          {[
            "React.js", "Next.js", "TypeScript", "JavaScript", "Component", "Props", "JSX",
            "Routing", "App Router", "Pages Router", "State Management", "Zustand", "Redux Toolkit",
            "Context API", "Hooks", "useState", "useEffect", "Data Fetching", "REST API",
            "SSR (Server-Side Rendering)", "CSR", "SSG", "ISR", "Hydration", "Tailwind CSS",
            "Core Web Vitals", "React.memo", "Testing Library", "Jest", "CI/CD",
          ].map((term) => (
            <span
              key={term}
              className="px-3 py-1 rounded-xl text-xs font-semibold bg-white dark:bg-gray-800 text-gray-800 dark:text-gray-200 border border-gray-200 dark:border-gray-700 shadow-sm"
            >
              {term}
            </span>
          ))}
        </div>
      </div>
    </DashboardShell>
  );
}
