import "./globals.css";
import { ReactNode } from "react";
import type { Metadata } from "next";

export const metadata: Metadata = {
  title: "AI Interview Assessment | TailAdmin Analytics Dashboard",
  description:
    "Automated AI interview video analysis with Whisper Speech-to-Text and NLP scoring for Frontend React & Next.js roles.",
};

export default function RootLayout({ children }: { children: ReactNode }) {
  return (
    <html lang="id" suppressHydrationWarning>
      <body className="antialiased min-h-screen">{children}</body>
    </html>
  );
}
