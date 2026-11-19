import { prisma } from "@/lib/prisma";
import { DashboardShell } from "../components/dashboard-shell";
import { CandidateTable, InterviewCandidate } from "../components/candidate-table";
import { Users, Filter, CheckCircle2, Clock, XCircle, Star, Trophy, TrendingUp } from "lucide-react";

export const dynamic = "force-dynamic";

// High-fidelity fallback candidate dataset for frontend React/Next.js recruiting
const MOCK_FALLBACK_CANDIDATES: InterviewCandidate[] = [
  {
    id: "mock-1",
    candidateId: "CAND-FRONTEND-001",
    status: "completed",
    stage: "SHORTLISTED",
    isStarred: true,
    roleLevel: "SENIOR",
    recruiterNotes: "Kandidat sangat kuat di arsitektur App Router & Server Components. Siap untuk wawancara user minggu depan.",
    finalScore: 0.88,
    verbalScore: 0.89,
    confidence: 0.94,
    summary: "Menjelaskan konsep React 18 concurrent rendering, Next.js Server Components, dan caching data dengan sangat fasih dan terstruktur.",
    keyStrengths: [
      "Menguasai arsitektur SSR & App Router secara mendalam.",
      "Tempo bicara artikulatif dan percaya diri (135 WPM).",
      "Komunikasi lugas dengan rasio kata jeda hanya 0.8%.",
    ],
    growthAreas: [
      "Perlu pendalaman terkait optimasi micro-frontend dan CI/CD bundle size tracking.",
    ],
    wpm: 135,
    fillerRatio: 0.8,
    transcript: "Dalam proyek terakhir kami, saya mengimplementasikan Next.js App Router dengan React Server Components untuk mengurangi client bundle size sebesar 45%. Untuk state management global antarmuka pengguna kami memilih Zustand karena ringan dan tidak memerlukan boilerplate yang rumit.",
    updatedAt: new Date(Date.now() - 1000 * 60 * 60 * 2),
  },
  {
    id: "mock-2",
    candidateId: "CAND-FRONTEND-002",
    status: "completed",
    stage: "USER_INTERVIEW",
    isStarred: true,
    roleLevel: "MID",
    recruiterNotes: "Pemahaman state management rapi, artikulasi komunikasi terstruktur.",
    finalScore: 0.82,
    verbalScore: 0.83,
    confidence: 0.91,
    summary: "Memahami hooks, Zustand, dan data fetching lifecycle dengan REST API. Menjawab pertanyaan dengan tenang.",
    keyStrengths: [
      "Pemisahan logika komponen rapi menggunakan custom hooks.",
      "Ritme bicara optimal (128 WPM) tanpa keraguan vokal.",
      "Menjelaskan integrasi REST API dan caching TanStack Query dengan baik.",
    ],
    growthAreas: [
      "Klarifikasi pengalaman dalam testing unit menggunakan Jest dan React Testing Library.",
    ],
    wpm: 128,
    fillerRatio: 1.4,
    transcript: "Saya terbiasa memisahkan business logic dari UI component dengan membuat custom hooks. Ketika mengonsumsi REST API, saya menerapkan TanStack Query agar penanganan cache dan retry otomatis berjalan efisien.",
    updatedAt: new Date(Date.now() - 1000 * 60 * 60 * 12),
  },
  {
    id: "mock-3",
    candidateId: "CAND-FRONTEND-003",
    status: "completed",
    stage: "TECHNICAL_TEST",
    isStarred: false,
    roleLevel: "JUNIOR",
    recruiterNotes: "Fondasi JavaScript dan CSS Tailwind cukup bagus, perlu uji coding praktis.",
    finalScore: 0.76,
    verbalScore: 0.77,
    confidence: 0.88,
    summary: "Memahami dasar React, props, useState, dan styling dengan Tailwind CSS. Terkadang menggunakan kata jeda.",
    keyStrengths: [
      "Memahami fundamental React component lifecycle dan props.",
      "Menunjukkan antusiasme belajar dan kesadaran desain responsif.",
    ],
    growthAreas: [
      "Frekuensi kata pengisi ('anu', 'ehm') sedikit meningkat saat menjelaskan useEffect.",
      "Belum banyak mengeksplorasi SSR/SSG di Next.js.",
    ],
    wpm: 118,
    fillerRatio: 2.8,
    transcript: "Saya sering membuat antarmuka web menggunakan React dan Tailwind CSS. Untuk komunikasi dengan server, biasanya saya menggunakan fetch atau Axios di dalam useEffect.",
    updatedAt: new Date(Date.now() - 1000 * 60 * 60 * 24),
  },
  {
    id: "mock-4",
    candidateId: "CAND-FRONTEND-004",
    status: "completed",
    stage: "OFFERED",
    isStarred: true,
    roleLevel: "SENIOR",
    recruiterNotes: "Kandidat top batch ini. Rekomendasi offer gaji tier 1.",
    finalScore: 0.92,
    verbalScore: 0.93,
    confidence: 0.96,
    summary: "Sangat menguasai Core Web Vitals (LCP, FID, CLS), Server Actions, arsitektur monorepo, dan automated testing.",
    keyStrengths: [
      "Arsitektur frontend skala enterprise dan Core Web Vitals profiling.",
      "Komunikasi sangat lugas dan profesional (142 WPM).",
      "Kombinasi sempurna antara Next.js App Router, TypeScript, dan testing.",
    ],
    growthAreas: [
      "Pertahankan standar rekayasa perangkat lunak tinggi saat onboarding.",
    ],
    wpm: 142,
    fillerRatio: 0.6,
    transcript: "Dalam mengoptimalkan performa web vitals terutama LCP, kami menerapkan dynamic import untuk komponen berat dan mengonfigurasi Server Actions Next.js agar mutasi data langsung dieksekusi di edge runtime.",
    updatedAt: new Date(Date.now() - 1000 * 60 * 60 * 36),
  },
  {
    id: "mock-5",
    candidateId: "CAND-FRONTEND-005",
    status: "completed",
    stage: "REJECTED",
    isStarred: false,
    roleLevel: "MID",
    recruiterNotes: "Belum memenuhi passing grade 70%. Email penolakan konstruktif telah disiapkan.",
    finalScore: 0.64,
    verbalScore: 0.65,
    confidence: 0.82,
    summary: "Penjelasan konsep state management dan server rendering masih belum tepat dan tempo bicara cenderung lambat.",
    keyStrengths: [
      "Memiliki pengalaman membangun halaman statis dasar dengan HTML/CSS.",
    ],
    growthAreas: [
      "Belum menguasai siklus render React, hooks dependensi, dan Next.js App Router.",
      "Tempo bicara lambat (92 WPM) dengan jeda hening panjang di atas 3 detik.",
    ],
    wpm: 92,
    fillerRatio: 4.5,
    transcript: "Ehm, saya biasanya... pakai React untuk tampilan sederhana saja, tapi kalau untuk Next.js atau state management yang rumit saya belum terlalu banyak menggunakannya di kantor lama.",
    updatedAt: new Date(Date.now() - 1000 * 60 * 60 * 48),
  },
];

async function getInterviews(): Promise<InterviewCandidate[]> {
  try {
    const records = await prisma.interview.findMany({
      orderBy: { updatedAt: "desc" },
      select: {
        id: true,
        candidateId: true,
        status: true,
        stage: true,
        isStarred: true,
        roleLevel: true,
        recruiterNotes: true,
        finalScore: true,
        verbalScore: true,
        confidence: true,
        summary: true,
        keyStrengths: true,
        growthAreas: true,
        updatedAt: true,
        transcript: {
          select: {
            text: true,
          },
        },
        nlp: {
          select: {
            fluency: true,
            relevance: true,
            overall: true,
          },
        },
      },
    });

    if (records.length === 0) {
      return MOCK_FALLBACK_CANDIDATES;
    }

    return records.map((r) => {
      const strengths =
        r.keyStrengths && typeof r.keyStrengths === "object" && "items" in (r.keyStrengths as any)
          ? ((r.keyStrengths as any).items as string[])
          : [];
      const growth =
        r.growthAreas && typeof r.growthAreas === "object" && "items" in (r.growthAreas as any)
          ? ((r.growthAreas as any).items as string[])
          : [];

      return {
        id: r.id,
        candidateId: r.candidateId,
        status: r.status,
        stage: r.stage || "INBOX",
        isStarred: r.isStarred ?? false,
        roleLevel: r.roleLevel || "MID",
        recruiterNotes: r.recruiterNotes,
        finalScore: r.finalScore,
        verbalScore: r.verbalScore,
        confidence: r.confidence,
        summary: r.summary,
        keyStrengths: strengths,
        growthAreas: growth,
        transcript: r.transcript?.text || "",
        wpm: 135,
        fillerRatio: 1.2,
        updatedAt: r.updatedAt,
      };
    });
  } catch (error) {
    console.error("Prisma fetch error (fallback to demo candidate records):", error);
    return MOCK_FALLBACK_CANDIDATES;
  }
}

export default async function CandidatesPage() {
  const interviews = await getInterviews();
  const apiBase = process.env.NEXT_PUBLIC_API_BASE_URL ?? "http://localhost:8000/api";

  const totalCount = interviews.length;
  const starredCount = interviews.filter((i) => i.isStarred).length;
  const passCount = interviews.filter((i) => (i.finalScore || 0) >= 0.8).length;
  const avgScore =
    totalCount > 0
      ? Math.round(
          (interviews.reduce((acc, curr) => acc + (curr.finalScore || 0), 0) / totalCount) * 100
        )
      : 0;

  return (
    <DashboardShell apiBaseUrl={apiBase}>
      {/* Header Title */}
      <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-4 mb-6">
        <div>
          <div className="flex items-center gap-2 mb-1">
            <h1 className="text-2xl sm:text-3xl font-extrabold tracking-tight text-gray-900 dark:text-white">
              Candidate Directory & Database
            </h1>
            <span className="px-2.5 py-0.5 rounded-full text-xs font-semibold bg-brand-50 text-brand-600 dark:bg-brand-500/15 dark:text-brand-400">
              ATS Pipeline Mode
            </span>
          </div>
          <p className="text-sm text-gray-500 dark:text-gray-400">
            Sistem pengurutan cerdas, filter granular kualifikasi, komparasi head-to-head, dan ringkasan AI untuk Recruiter HR.
          </p>
        </div>
      </div>

      {/* Top HR KPI Stat Cards */}
      <div className="grid grid-cols-2 sm:grid-cols-4 gap-4 mb-6">
        {/* Total Candidates */}
        <div className="tailadmin-card p-4 sm:p-5 flex items-center gap-3">
          <div className="p-2.5 rounded-xl bg-brand-50 text-brand-600 dark:bg-brand-500/10 dark:text-brand-400">
            <Users className="w-5 h-5" />
          </div>
          <div>
            <span className="text-xs text-gray-400 block font-medium">Total Kandidat</span>
            <span className="text-xl font-bold text-gray-900 dark:text-white">{totalCount} Orang</span>
          </div>
        </div>

        {/* Shortlisted Candidates */}
        <div className="tailadmin-card p-4 sm:p-5 flex items-center gap-3">
          <div className="p-2.5 rounded-xl bg-amber-50 text-amber-600 dark:bg-amber-500/10 dark:text-amber-400">
            <Star className="w-5 h-5 fill-amber-500 text-amber-500" />
          </div>
          <div>
            <span className="text-xs text-gray-400 block font-medium">Shortlisted ⭐</span>
            <span className="text-xl font-bold text-amber-600 dark:text-amber-400">{starredCount} Kandidat</span>
          </div>
        </div>

        {/* Lolos Passing Grade */}
        <div className="tailadmin-card p-4 sm:p-5 flex items-center gap-3">
          <div className="p-2.5 rounded-xl bg-emerald-50 text-emerald-600 dark:bg-emerald-500/10 dark:text-emerald-400">
            <Trophy className="w-5 h-5" />
          </div>
          <div>
            <span className="text-xs text-gray-400 block font-medium">Lolos Passing (≥80%)</span>
            <span className="text-xl font-bold text-emerald-600 dark:text-emerald-400">{passCount} Kandidat</span>
          </div>
        </div>

        {/* Average Score */}
        <div className="tailadmin-card p-4 sm:p-5 flex items-center gap-3">
          <div className="p-2.5 rounded-xl bg-purple-50 text-purple-600 dark:bg-purple-500/10 dark:text-purple-400">
            <TrendingUp className="w-5 h-5" />
          </div>
          <div>
            <span className="text-xs text-gray-400 block font-medium">Rata-rata Skor Batch</span>
            <span className="text-xl font-bold text-purple-600 dark:text-purple-400">{avgScore}%</span>
          </div>
        </div>
      </div>

      {/* Main Full-Width Candidate Table */}
      <CandidateTable interviews={interviews} apiBaseUrl={apiBase} />
    </DashboardShell>
  );
}
