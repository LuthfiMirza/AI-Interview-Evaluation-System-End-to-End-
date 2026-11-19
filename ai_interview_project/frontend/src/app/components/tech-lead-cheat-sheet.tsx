'use client';

import { useState } from "react";
import { HelpCircle, Copy, Check, Printer, Sparkles, BookOpen, ChevronRight, CheckCircle2 } from "lucide-react";

export type TechLeadQuestion = {
  id: number;
  category: string;
  question: string;
  contextForTechLead: string;
  expectedAnswer: string;
};

type TechLeadCheatSheetProps = {
  candidateId: string;
  roleLevel: string;
  finalScore: number | null;
  keyStrengths?: string[];
  growthAreas?: string[];
  missingConcepts?: string[];
};

export function generateTechLeadQuestions(
  roleLevel: string,
  growthAreas: string[] = [],
  missingConcepts: string[] = []
): TechLeadQuestion[] {
  const isSenior = roleLevel.toUpperCase() === "SENIOR";
  const isJunior = roleLevel.toUpperCase() === "JUNIOR";

  if (isSenior) {
    return [
      {
        id: 1,
        category: "Arsitektur & Server Components",
        question:
          "Dalam arsitektur Next.js App Router skala enterprise, bagaimana pendekatan Anda dalam memisahkan batas (boundary) Server vs Client Components untuk mencegah waterfall fetching dan menjaga bundle size seminimal mungkin?",
        contextForTechLead:
          "Kandidat menunjukkan pemahaman konseptual yang baik, namun perlu diuji pada implementasi nested layout dan streaming Suspense nyata di level Senior.",
        expectedAnswer:
          "Menjelaskan pemindahan directive 'use client' ke leaf components, pemanfaatan parallel routes / slots, dan composability server components as props.",
      },
      {
        id: 2,
        category: "State Management & Cache Invalidation",
        question:
          "Kapan Anda merekomendasikan Zustand dibanding Server Actions atau TanStack Query, dan bagaimana strategi Anda menangani sinkronisasi data optimistik (optimistic UI update) saat mutasi server gagal?",
        contextForTechLead:
          "Verifikasi apakah kandidat memahami perbedaan mendasar antara transient client state dengan synchronized server state cache.",
        expectedAnswer:
          "Menyebutkan bahwa Zustand fokus pada UI state lokal, sedangkan mutasi server ditangani dengan revalidateTag/revalidatePath dan rollback manual pada rollback state.",
      },
      {
        id: 3,
        category: "Web Vitals & Performance Profiling",
        question:
          "Ketika aplikasi Next.js mengalami penurunan skor LCP (Largest Contentful Paint) dan CLS di production, langkah profiling apa yang Anda ambil dengan Chrome DevTools, dan optimasi apa yang diterapkan pada font/image loading?",
        contextForTechLead:
          "Evaluasi ketelitian kandidat Senior dalam memitigasi layout shift dan performa rendering edge.",
        expectedAnswer:
          "Menjelaskan next/image dengan priority attribute, next/font dengan zero layout shift, resource hints preload/preconnect, dan CDN edge caching.",
      },
    ];
  }

  if (isJunior) {
    return [
      {
        id: 1,
        category: "Fundamental Component & Props",
        question:
          "Bisa jelaskan perbedaan antara state dan props di React, dan bagaimana cara Anda menghindari masalah 'props drilling' ketika beberapa komponen anak membutuhkan data yang sama?",
        contextForTechLead:
          "Menilai apakah kandidat Junior menguasai dasar unidirectional data flow dan kapan harus menggunakan context atau composition.",
        expectedAnswer:
          "Menjelaskan bahwa props bersifat read-only dari parent, sedangkan state dikelola secara internal; props drilling dihindari dengan composition atau simple Context/Zustand.",
      },
      {
        id: 2,
        category: "Hooks Lifecycle & Side Effects",
        question:
          "Apa fungsi dependency array pada useEffect, dan apa yang akan terjadi jika Anda memanggil pembaruan state di dalam useEffect tanpa dependency array?",
        contextForTechLead:
          "Kandidat Junior sering membuat infinite loop atau memory leak pada useEffect; ini menguji pemahaman siklus hidup render.",
        expectedAnswer:
          "Menjelaskan potensi infinite re-render loop dan pentingnya cleanup function untuk event listeners atau subscription timer.",
      },
      {
        id: 3,
        category: "Data Fetching & Error Handling",
        question:
          "Ketika melakukan fetch data dari REST API ke backend, bagaimana Anda menangani kondisi loading, sukses, dan error di antarmuka pengguna?",
        contextForTechLead:
          "Menguji kesadaran kandidat terhadap User Experience (UX) saat network lambat atau server mengembalikan error 500.",
        expectedAnswer:
          "Menjelaskan pemisahan state loading skeleton, penanganan try-catch / async-await, dan penyampaian pesan error yang ramah pengguna.",
      },
    ];
  }

  // Default: Mid Level
  return [
    {
      id: 1,
      category: "Custom Hooks & Modularitas",
      question:
        "Bagaimana Anda merancang Custom Hook yang memisahkan logika data fetching dan state handling dari komponen visual, dan bagaimana Anda memastikan hook tersebut mudah di-test?",
      contextForTechLead:
        "Memverifikasi kebiasaan clean code dan separation of concerns kandidat Mid-level.",
      expectedAnswer:
        "Menyebutkan enkapsulasi logic di hook terpisah, pengembalian state terdefinisi (data, isLoading, isError), dan testing hook menggunakan renderHook dari React Testing Library.",
    },
    {
      id: 2,
      category: "Next.js SSR vs CSR Optimization",
      question:
        "Dalam proyek Next.js, kapan Anda memilih Server-Side Rendering (SSR/Server Component) dibanding Client-Side Rendering (CSR), dan bagaimana dampaknya terhadap performa First Contentful Paint (FCP)?",
      contextForTechLead:
        "Menilai apakah kandidat memahami kapan menggunakan browser API vs kemampuan komputasi server Node.js.",
      expectedAnswer:
        "Menjelaskan bahwa konten publik/SEO membutuhkan SSR untuk HTML siap saji yang mempercepat FCP, sedangkan halaman dashboard interaktif dengan auth cookie dapat memanfaatkan dynamic client rendering.",
    },
    {
      id: 3,
      category: "Automated Testing & Kualitas Kode",
      question:
        "Bagaimana strategi Anda dalam menulis unit test untuk form interaktif menggunakan React Testing Library? Apakah Anda lebih menyukai menguji implementation detail atau user behavior?",
      contextForTechLead:
        "Memastikan kandidat tidak hanya bisa membuat UI, tetapi juga menjaga stabilitas kode tim dari regresi.",
      expectedAnswer:
        "Menjelaskan prinsip testing user behavior (e.g. fireEvent / userEvent.click, findByRole) daripada mengecek state internal komponen.",
    },
  ];
}

export function TechLeadCheatSheet({
  candidateId,
  roleLevel,
  finalScore,
  keyStrengths = [],
  growthAreas = [],
  missingConcepts = [],
}: TechLeadCheatSheetProps) {
  const [copied, setCopied] = useState(false);
  const questions = generateTechLeadQuestions(roleLevel, growthAreas, missingConcepts);

  const scorePct = finalScore !== null ? `${Math.round(finalScore * 100)}%` : "N/A";

  const handleCopyCheatSheet = async () => {
    const textLines = [
      `=====================================================`,
      `PANDUAN WAWANCARA TEKNIS (TECH LEAD CHEAT-SHEET)`,
      `Kandidat: ${candidateId} | Level: ${roleLevel} | Skor AI: ${scorePct}`,
      `=====================================================`,
      ``,
      `🎯 REKOMENDASI 3 PERTANYAAN KHUSUS UNTUK TECH LEAD:`,
      ``,
    ];

    questions.forEach((q, i) => {
      textLines.push(`[${i + 1}] Kategori: ${q.category}`);
      textLines.push(`PERTANYAAN: "${q.question}"`);
      textLines.push(`Konteks Analisis AI: ${q.contextForTechLead}`);
      textLines.push(`Ekspektasi Jawaban Ideal: ${q.expectedAnswer}`);
      textLines.push(``);
    });

    textLines.push(`Catatan untuk Tech Lead: Pertanyaan ini dibuat otomatis berdasarkan hasil transkripsi dan konsep yang perlu diklarifikasi dari video wawancara AI.`);

    try {
      await navigator.clipboard.writeText(textLines.join("\n"));
      setCopied(true);
      setTimeout(() => setCopied(false), 2500);
    } catch (err) {
      console.error("Gagal menyalin cheat-sheet:", err);
    }
  };

  return (
    <div className="space-y-4">
      {/* Top Banner */}
      <div className="p-4 rounded-xl bg-purple-50/70 dark:bg-purple-500/10 border border-purple-200/60 dark:border-purple-800/60 flex items-start justify-between gap-3 text-xs">
        <div className="flex items-start gap-2.5">
          <Sparkles className="w-4 h-4 text-purple-600 dark:text-purple-400 mt-0.5 shrink-0" />
          <div>
            <span className="font-bold text-gray-900 dark:text-white block mb-0.5">
              Panduan Tanya Khusus Tech Lead / Hiring Manager
            </span>
            <p className="text-gray-600 dark:text-gray-300 text-[11px] leading-relaxed">
              Dihasilkan otomatis oleh AI berdasarkan titik lemah (*gap*) kandidat pada rekaman wawancara. Siap di-forward ke tim engineering untuk sesi *live interview*.
            </p>
          </div>
        </div>

        <button
          onClick={handleCopyCheatSheet}
          className="inline-flex items-center gap-1.5 px-3 py-1.5 rounded-lg font-semibold bg-white dark:bg-gray-800 border border-purple-200 dark:border-purple-700 text-purple-700 dark:text-purple-300 hover:bg-purple-50 transition-colors shrink-0 shadow-sm"
        >
          {copied ? (
            <>
              <Check className="w-3.5 h-3.5 text-emerald-500" />
              <span className="text-emerald-600">Tersalin!</span>
            </>
          ) : (
            <>
              <Copy className="w-3.5 h-3.5" />
              Salin Panduan
            </>
          )}
        </button>
      </div>

      {/* 3 Questions Cards */}
      <div className="space-y-3">
        {questions.map((q, idx) => (
          <div
            key={q.id}
            className="p-4 rounded-xl border border-gray-200 dark:border-gray-800 bg-white dark:bg-gray-900/50 space-y-2.5 text-xs shadow-sm"
          >
            <div className="flex items-center justify-between">
              <span className="px-2 py-0.5 rounded-md font-bold text-[10px] bg-brand-50 text-brand-600 dark:bg-brand-500/15 dark:text-brand-400">
                Pertanyaan {idx + 1} • {q.category}
              </span>
            </div>

            <p className="font-bold text-gray-900 dark:text-white text-xs leading-relaxed">
              &ldquo;{q.question}&rdquo;
            </p>

            <div className="pt-2 border-t border-gray-100 dark:border-gray-800 space-y-1.5 text-[11px]">
              <div className="flex items-start gap-1.5 text-amber-700 dark:text-amber-400">
                <span className="font-semibold shrink-0">🔍 Alasan / Konteks AI:</span>
                <span className="text-gray-600 dark:text-gray-300">{q.contextForTechLead}</span>
              </div>

              <div className="flex items-start gap-1.5 text-emerald-700 dark:text-emerald-400">
                <span className="font-semibold shrink-0">🎯 Jawaban Ideal:</span>
                <span className="text-gray-600 dark:text-gray-300">{q.expectedAnswer}</span>
              </div>
            </div>
          </div>
        ))}
      </div>
    </div>
  );
}
