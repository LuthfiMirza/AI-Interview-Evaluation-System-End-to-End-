'use client';

import { useState } from "react";
import { Mail, Copy, Check, X, Sparkles, Send } from "lucide-react";

export type EmailCandidateData = {
  candidateId: string;
  roleLevel: string;
  finalScore: number | null;
  verbalScore: number | null;
  keyStrengths?: string[];
  growthAreas?: string[];
};

type RecruiterEmailModalProps = {
  candidate: EmailCandidateData | null;
  isOpen: boolean;
  onClose: () => void;
};

export function RecruiterEmailModal({ candidate, isOpen, onClose }: RecruiterEmailModalProps) {
  const [activeTab, setActiveTab] = useState<"INVITE" | "REJECT">("INVITE");
  const [copied, setCopied] = useState(false);

  if (!isOpen || !candidate) return null;

  const scorePct = candidate.finalScore !== null ? `${Math.round(candidate.finalScore * 100)}%` : "N/A";
  const strengthsList =
    candidate.keyStrengths && candidate.keyStrengths.length > 0
      ? candidate.keyStrengths.map((s) => `• ${s}`).join("\n")
      : "• Pemahaman konseptual React & Next.js yang memadai\n• Komunikasi teknis yang terstruktur";

  const inviteSubject = `Undangan Technical Interview — Frontend Engineer (${candidate.roleLevel}) | ${candidate.candidateId}`;
  const inviteBody = `Halo ${candidate.candidateId},

Terima kasih telah menyelesaikan sesi evaluasi video wawancara AI untuk posisi Frontend Engineer (${candidate.roleLevel}).

Berdasarkan hasil analisis sistem asesmen AI kami, Anda meraih skor evaluasi keseluruhan ${scorePct} dengan poin keunggulan utama sebagai berikut:
${strengthsList}

Kami sangat terkesan dengan pemahaman teknis dan artikulasi respon Anda. Oleh karena itu, kami ingin mengundang Anda ke tahap selanjutnya: Sesi Technical Live Interview bersama Tim Engineering Lead kami.

Rincian Sesi Lanjutan:
- Platform: Google Meet
- Durasi: 45 Menit (Code Review, Architecture Deep-dive & Q&A)
- Jadwal: Silakan pilih waktu yang sesuai melalui tautan berikut [Tautan Kalender / Calendly].

Mohon konfirmasi ketersediaan Anda dengan membalas email ini. Kami menantikan sesi diskusi mendalam dengan Anda!

Salam hangat,
Tim Rekrutmen & Talent Acquisition`;

  const rejectSubject = `Pembaruan Proses Rekrutmen — Frontend Engineer (${candidate.roleLevel}) | ${candidate.candidateId}`;
  const rejectBody = `Halo ${candidate.candidateId},

Terima kasih atas waktu dan dedikasi yang Anda luangkan dalam mengikuti sesi evaluasi wawancara AI untuk posisi Frontend Engineer (${candidate.roleLevel}).

Tim kami sangat mengapresiasi partisipasi Anda. Melalui evaluasi video yang telah berlangsung, sistem kami mencatat keunggulan dalam:
${strengthsList}

Namun demikian, setelah membandingkan dengan kebutuhan spesifik dan kuota posisi saat ini, kami memutuskan untuk belum dapat melanjutkan proses ke tahapan berikutnya.

Profil Anda akan tetap tersimpan di dalam database talent pool kami untuk peluang karir di masa mendatang yang relevan dengan keahlian Anda. Kami mendoakan kesuksesan terbaik dalam perjalanan karir profesional Anda.

Salam hormat,
Tim Rekrutmen & Talent Acquisition`;

  const currentSubject = activeTab === "INVITE" ? inviteSubject : rejectSubject;
  const currentBody = activeTab === "INVITE" ? inviteBody : rejectBody;

  const handleCopy = async () => {
    try {
      await navigator.clipboard.writeText(`Subject: ${currentSubject}\n\n${currentBody}`);
      setCopied(true);
      setTimeout(() => setCopied(false), 2500);
    } catch (err) {
      console.error("Gagal menyalin email:", err);
    }
  };

  const handleMailto = () => {
    const encodedSubject = encodeURIComponent(currentSubject);
    const encodedBody = encodeURIComponent(currentBody);
    window.open(`mailto:?subject=${encodedSubject}&body=${encodedBody}`, "_blank");
  };

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center p-4 bg-gray-900/60 backdrop-blur-sm animate-in fade-in duration-200">
      <div className="tailadmin-card max-w-2xl w-full p-6 bg-white dark:bg-[#101828] shadow-2xl overflow-hidden border border-gray-200 dark:border-gray-800">
        {/* Header */}
        <div className="flex items-center justify-between pb-4 mb-4 border-b border-gray-100 dark:border-gray-800">
          <div className="flex items-center gap-3">
            <div className="p-2.5 rounded-xl bg-brand-50 text-brand-600 dark:bg-brand-500/10 dark:text-brand-400">
              <Mail className="w-5 h-5" />
            </div>
            <div>
              <h3 className="text-base font-bold text-gray-900 dark:text-white flex items-center gap-2">
                Automated Recruiter Email Composer
                <span className="px-2 py-0.5 rounded-full text-[10px] font-semibold bg-emerald-50 text-emerald-600 dark:bg-emerald-500/10 dark:text-emerald-400">
                  AI-Personalized
                </span>
              </h3>
              <p className="text-xs text-gray-500 dark:text-gray-400">
                Kandidat: <span className="font-semibold text-gray-700 dark:text-gray-200">{candidate.candidateId}</span> ({candidate.roleLevel}) • Skor AI: <span className="font-bold text-brand-600">{scorePct}</span>
              </p>
            </div>
          </div>
          <button
            onClick={onClose}
            className="p-1.5 rounded-lg text-gray-400 hover:text-gray-600 dark:hover:text-gray-300 hover:bg-gray-100 dark:hover:bg-gray-800 transition-colors"
          >
            <X className="w-5 h-5" />
          </button>
        </div>

        {/* Tab Selection */}
        <div className="flex gap-2 p-1 bg-gray-100 dark:bg-gray-800/80 rounded-xl mb-4 text-xs font-semibold">
          <button
            onClick={() => setActiveTab("INVITE")}
            className={`flex-1 py-2 px-3 rounded-lg transition-all flex items-center justify-center gap-2 ${
              activeTab === "INVITE"
                ? "bg-white dark:bg-gray-700 text-emerald-600 dark:text-emerald-400 shadow-sm"
                : "text-gray-500 hover:text-gray-900 dark:text-gray-400"
            }`}
          >
            <Sparkles className="w-3.5 h-3.5" />
            Lolos ke Sesi Tech Interview
          </button>
          <button
            onClick={() => setActiveTab("REJECT")}
            className={`flex-1 py-2 px-3 rounded-lg transition-all flex items-center justify-center gap-2 ${
              activeTab === "REJECT"
                ? "bg-white dark:bg-gray-700 text-rose-600 dark:text-rose-400 shadow-sm"
                : "text-gray-500 hover:text-gray-900 dark:text-gray-400"
            }`}
          >
            <X className="w-3.5 h-3.5" />
            Penolakan Sopan & Feedback Konstruktif
          </button>
        </div>

        {/* Email Preview Subject */}
        <div className="space-y-3 mb-4 text-xs">
          <div>
            <label className="block text-[11px] font-semibold text-gray-500 dark:text-gray-400 uppercase tracking-wider mb-1">
              Subjek Email
            </label>
            <input
              type="text"
              readOnly
              value={currentSubject}
              className="w-full px-3 py-2 text-xs font-medium rounded-xl bg-gray-50 dark:bg-gray-900/60 border border-gray-200 dark:border-gray-800 text-gray-800 dark:text-gray-200 focus:outline-none"
            />
          </div>

          <div>
            <label className="block text-[11px] font-semibold text-gray-500 dark:text-gray-400 uppercase tracking-wider mb-1">
              Isi Pesan Email (Telah Dipersonalisasi AI)
            </label>
            <textarea
              rows={10}
              readOnly
              value={currentBody}
              className="w-full p-3 font-mono text-xs rounded-xl bg-gray-50 dark:bg-gray-900/60 border border-gray-200 dark:border-gray-800 text-gray-800 dark:text-gray-200 focus:outline-none leading-relaxed resize-none"
            />
          </div>
        </div>

        {/* Footer Actions */}
        <div className="flex items-center justify-between pt-3 border-t border-gray-100 dark:border-gray-800 text-xs">
          <p className="text-[11px] text-gray-400">
            💡 Teks otomatis menyertakan poin kelebihan (*strengths*) dari pipeline Whisper & NLP.
          </p>
          <div className="flex items-center gap-2">
            <button
              onClick={handleCopy}
              className="inline-flex items-center gap-1.5 px-3.5 py-2 rounded-xl font-semibold border border-gray-200 dark:border-gray-700 hover:bg-gray-50 dark:hover:bg-gray-800 text-gray-700 dark:text-gray-200 transition-colors"
            >
              {copied ? (
                <>
                  <Check className="w-3.5 h-3.5 text-emerald-500" />
                  <span className="text-emerald-600">Tersalin!</span>
                </>
              ) : (
                <>
                  <Copy className="w-3.5 h-3.5" />
                  Salin Teks
                </>
              )}
            </button>
            <button
              onClick={handleMailto}
              className="inline-flex items-center gap-1.5 px-4 py-2 rounded-xl font-semibold text-white bg-brand-500 hover:bg-brand-600 transition-colors shadow-sm"
            >
              <Send className="w-3.5 h-3.5" />
              Buka di Mail App
            </button>
          </div>
        </div>
      </div>
    </div>
  );
}
