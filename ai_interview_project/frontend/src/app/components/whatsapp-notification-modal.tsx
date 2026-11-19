'use client';

import { useState } from "react";
import { MessageSquare, Phone, Copy, Check, ExternalLink, X, Sparkles } from "lucide-react";

export type WhatsAppCandidateData = {
  candidateId: string;
  roleLevel: string;
  finalScore: number | null;
  keyStrengths?: string[];
  phone?: string;
};

type WhatsAppModalProps = {
  candidate: WhatsAppCandidateData | null;
  isOpen: boolean;
  onClose: () => void;
};

export function WhatsAppNotificationModal({ candidate, isOpen, onClose }: WhatsAppModalProps) {
  const [phoneNumber, setPhoneNumber] = useState(candidate?.phone || "081234567890");
  const [activeTab, setActiveTab] = useState<"INVITE" | "REJECT">("INVITE");
  const [copied, setCopied] = useState(false);

  if (!isOpen || !candidate) return null;

  const scorePct = candidate.finalScore !== null ? `${Math.round(candidate.finalScore * 100)}%` : "N/A";
  const strengthsText =
    candidate.keyStrengths && candidate.keyStrengths.length > 0
      ? candidate.keyStrengths.map((s) => `• ${s}`).join("\n")
      : "• Pemahaman arsitektur React & Next.js yang terstruktur\n• Komunikasi teknis yang lugas";

  // Sanitize phone number for wa.me link (convert 08xx to 628xx)
  const cleanPhone = (raw: string) => {
    let p = raw.replace(/\D/g, "");
    if (p.startsWith("0")) {
      p = "62" + p.slice(1);
    }
    return p;
  };

  const inviteMessage = `*Halo Sdr/i ${candidate.candidateId},*

Selamat! Berdasarkan hasil evaluasi video wawancara AI kami untuk posisi *Frontend Engineer (${candidate.roleLevel})*, Anda dinyatakan *LOLOS* dengan skor akhir *${scorePct}*.

*Poin Keunggulan Utama:*
${strengthsText}

Kami mengundang Anda ke tahapan berikutnya: *Sesi Technical Live Interview* bersama Tim Engineering Lead kami (45 Menit, Google Meet).

Mohon konfirmasi kesediaan Anda dengan membalas pesan WhatsApp ini, atau pilih jadwal yang cocok melalui: [Tautan Jadwal / Calendly].

Terima kasih dan sampai jumpa di sesi wawancara!

Salam hangat,
*Tim Rekrutmen & Talent Acquisition*`;

  const rejectMessage = `*Halo Sdr/i ${candidate.candidateId},*

Terima kasih atas partisipasi Anda dalam sesi evaluasi wawancara AI untuk posisi *Frontend Engineer (${candidate.roleLevel})*.

Tim kami sangat mengapresiasi waktu dan pemaparan Anda, khususnya dalam:
${strengthsText}

Namun demikian, sehubungan dengan keterbatasan kuota posisi saat ini, kami memutuskan belum dapat melanjutkan proses ke tahapan berikutnya.

Data profil Anda tetap kami simpan di talent pool kami untuk peluang karir di masa depan. Sukses selalu untuk perjalanan karir profesional Anda!

Salam hormat,
*Tim Rekrutmen & Talent Acquisition*`;

  const currentMessage = activeTab === "INVITE" ? inviteMessage : rejectMessage;

  const handleCopy = async () => {
    try {
      await navigator.clipboard.writeText(currentMessage);
      setCopied(true);
      setTimeout(() => setCopied(false), 2500);
    } catch (err) {
      console.error("Gagal menyalin pesan WA:", err);
    }
  };

  const handleOpenWhatsApp = () => {
    const formattedPhone = cleanPhone(phoneNumber);
    const encoded = encodeURIComponent(currentMessage);
    const url = formattedPhone
      ? `https://wa.me/${formattedPhone}?text=${encoded}`
      : `https://wa.me/?text=${encoded}`;
    window.open(url, "_blank");
  };

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center p-4 bg-gray-900/60 backdrop-blur-sm animate-in fade-in duration-200">
      <div className="tailadmin-card max-w-xl w-full p-6 bg-white dark:bg-[#101828] shadow-2xl overflow-hidden border border-gray-200 dark:border-gray-800">
        {/* Header */}
        <div className="flex items-center justify-between pb-4 mb-4 border-b border-gray-100 dark:border-gray-800">
          <div className="flex items-center gap-3">
            <div className="p-2.5 rounded-xl bg-emerald-50 text-emerald-600 dark:bg-emerald-500/15 dark:text-emerald-400">
              <MessageSquare className="w-5 h-5" />
            </div>
            <div>
              <h3 className="text-base font-bold text-gray-900 dark:text-white flex items-center gap-2">
                1-Click WhatsApp Quick Notification
                <span className="px-2 py-0.5 rounded-full text-[10px] font-bold bg-emerald-50 text-emerald-700 dark:bg-emerald-500/20 dark:text-emerald-400">
                  Respon Cepat
                </span>
              </h3>
              <p className="text-xs text-gray-400">
                Kirim pesan instan terformat langsung ke WhatsApp kandidat {candidate.candidateId}.
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

        {/* Phone Input & Tab Selector */}
        <div className="space-y-3 mb-4 text-xs">
          <div>
            <label className="block text-[11px] font-semibold text-gray-500 dark:text-gray-400 uppercase tracking-wider mb-1">
              Nomor WhatsApp Kandidat
            </label>
            <div className="relative">
              <Phone className="w-4 h-4 text-gray-400 absolute left-3 top-1/2 -translate-y-1/2" />
              <input
                type="text"
                value={phoneNumber}
                onChange={(e) => setPhoneNumber(e.target.value)}
                placeholder="e.g. 08123456789 atau +628123456789"
                className="w-full pl-9 pr-3 py-2 text-xs font-semibold rounded-xl bg-gray-50 dark:bg-gray-900/60 border border-gray-200 dark:border-gray-800 text-gray-900 dark:text-white focus:outline-none focus:ring-2 focus:ring-emerald-500"
              />
            </div>
          </div>

          <div className="flex gap-2 p-1 bg-gray-100 dark:bg-gray-800/80 rounded-xl font-semibold">
            <button
              onClick={() => setActiveTab("INVITE")}
              className={`flex-1 py-1.5 px-3 rounded-lg transition-all flex items-center justify-center gap-1.5 ${
                activeTab === "INVITE"
                  ? "bg-white dark:bg-gray-700 text-emerald-600 dark:text-emerald-400 shadow-sm"
                  : "text-gray-500 hover:text-gray-900 dark:text-gray-400"
              }`}
            >
              <Sparkles className="w-3 h-3" />
              Undangan Interview
            </button>
            <button
              onClick={() => setActiveTab("REJECT")}
              className={`flex-1 py-1.5 px-3 rounded-lg transition-all flex items-center justify-center gap-1.5 ${
                activeTab === "REJECT"
                  ? "bg-white dark:bg-gray-700 text-rose-600 dark:text-rose-400 shadow-sm"
                  : "text-gray-500 hover:text-gray-900 dark:text-gray-400"
              }`}
            >
              Penolakan Sopan
            </button>
          </div>

          {/* Message Preview */}
          <div>
            <label className="block text-[11px] font-semibold text-gray-500 dark:text-gray-400 uppercase tracking-wider mb-1">
              Pratinjau Pesan WhatsApp (Format Bold *...*)
            </label>
            <textarea
              rows={8}
              readOnly
              value={currentMessage}
              className="w-full p-3 font-mono text-xs rounded-xl bg-gray-50 dark:bg-gray-900/60 border border-gray-200 dark:border-gray-800 text-gray-800 dark:text-gray-200 focus:outline-none leading-relaxed resize-none"
            />
          </div>
        </div>

        {/* Footer Actions */}
        <div className="flex items-center justify-between pt-3 border-t border-gray-100 dark:border-gray-800 text-xs">
          <p className="text-[11px] text-gray-400">
            💬 Menggunakan standar protokol resmi <span className="font-mono">wa.me</span>.
          </p>
          <div className="flex items-center gap-2">
            <button
              onClick={handleCopy}
              className="inline-flex items-center gap-1.5 px-3 py-2 rounded-xl font-semibold border border-gray-200 dark:border-gray-700 hover:bg-gray-50 dark:hover:bg-gray-800 text-gray-700 dark:text-gray-200 transition-colors"
            >
              {copied ? (
                <>
                  <Check className="w-3.5 h-3.5 text-emerald-500" />
                  <span className="text-emerald-600">Tersalin!</span>
                </>
              ) : (
                <>
                  <Copy className="w-3.5 h-3.5" />
                  Salin Pesan
                </>
              )}
            </button>

            <button
              onClick={handleOpenWhatsApp}
              className="inline-flex items-center gap-1.5 px-4 py-2 rounded-xl font-semibold text-white bg-emerald-600 hover:bg-emerald-500 transition-colors shadow-sm"
            >
              <ExternalLink className="w-3.5 h-3.5" />
              Buka WhatsApp Web
            </button>
          </div>
        </div>
      </div>
    </div>
  );
}
