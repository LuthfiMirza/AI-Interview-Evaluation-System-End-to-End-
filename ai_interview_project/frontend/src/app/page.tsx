import { formatDistanceToNow } from "date-fns";
import Link from "next/link";

import { prisma } from "@/lib/prisma";
import { ReportCard } from "./components/report-card";
import { UploadPanel } from "./components/upload-panel";

type InterviewRow = {
  id: string;
  candidateId: string;
  status: string;
  finalScore: number | null;
  verbalScore: number | null;
  nonVerbalScore: number | null;
  cheatingScore: number | null;
  confidence: number | null;
  summary: string | null;
  updatedAt: Date;
};

async function getInterviews(): Promise<InterviewRow[]> {
  return prisma.interview.findMany({
    orderBy: { updatedAt: "desc" },
    select: {
      id: true,
      candidateId: true,
      status: true,
      finalScore: true,
      verbalScore: true,
      nonVerbalScore: true,
      cheatingScore: true,
      confidence: true,
      summary: true,
      updatedAt: true,
    },
  });
}

const statusClass = (status: string) => {
  switch (status) {
    case "completed":
      return "status-pill status-completed";
    case "failed":
      return "status-pill status-failed";
    default:
      return "status-pill status-processing";
  }
};

export default async function DashboardPage() {
  const interviews = await getInterviews();
  const latestInterview = interviews[0] ?? null;
  const apiBase =
    process.env.NEXT_PUBLIC_API_BASE_URL ?? "http://localhost:8000/api";

  return (
    <div className="container">
      <section className="hero">
        <p className="step-label">AI Interview Assessment – MVP</p>
        <h1>AI Interview Assessment – MVP</h1>
        <p>
          Upload video interview → Transcribe (EN/ID) → Optional analysis
          (diarization, non-verbal, cheating) → Download report.
        </p>
        <p className="hero-subtext">
          Backend docs:{" "}
          <Link href="http://localhost:8000/docs">http://localhost:8000/docs</Link>{" "}
          • API Base URL <code>{apiBase}</code>
        </p>
      </section>

      <div className="steps">
        <section className="card">
          <div className="step-label">1) Upload Video</div>
          <h3>Upload Video</h3>
          <p className="muted">MP4 · MOV · MKV. Max ~1 GB (configurable).</p>
          <UploadPanel apiBaseUrl={apiBase} />
        </section>

        <section className="card">
          <div className="step-label">2) Processing Pipeline</div>
          <h3>Transcribe & Analyse</h3>
          <p className="muted">
            Once uploaded, the backend performs the following steps automatically:
          </p>
          <ol className="pipeline-list">
            <li>Extract & denoise audio, resample to 16 kHz mono.</li>
            <li>Whisper STT (medium.en) + confidence scoring.</li>
            <li>
              NLP scoring (relevance, fluency, summary) + vision metrics (eye
              contact, cheating cues).
            </li>
            <li>Aggregate into a unified report and persist via PostgreSQL.</li>
          </ol>
        </section>

        <section className="card">
          <div className="step-label">3) Results & Report</div>
          <h3>Results & Report</h3>
          <p className="muted">Transcription, metrics, and export-ready notes.</p>
          <ReportCard interview={latestInterview} />
        </section>

        <section className="card">
          <div className="step-label">4) Interview Log</div>
          <h3>Interview Log</h3>
          <p className="muted">
            All processed interviews stored via Prisma ORM. Refresh after jobs
            complete to see the latest status.
          </p>
          <div className="table-wrapper">
            <table>
              <thead>
                <tr>
                  <th>Candidate</th>
                  <th>Status</th>
                  <th>Final</th>
                  <th>Verbal</th>
                  <th>Non-Verbal</th>
                  <th>Cheating</th>
                  <th>Confidence</th>
                  <th>Updated</th>
                  <th>Summary</th>
                </tr>
              </thead>
              <tbody>
                {interviews.length === 0 && (
                  <tr>
                    <td colSpan={9} style={{ textAlign: "center", padding: 28 }}>
                      No interviews stored yet. Upload a video and click{" "}
                      <strong>Process</strong>.
                    </td>
                  </tr>
                )}
                {interviews.map((row) => (
                  <tr key={row.id}>
                    <td>{row.candidateId}</td>
                    <td>
                      <span className={statusClass(row.status)}>
                        {row.status}
                      </span>
                    </td>
                    <td>{row.finalScore?.toFixed(2) ?? "—"}</td>
                    <td>{row.verbalScore?.toFixed(2) ?? "—"}</td>
                    <td>{row.nonVerbalScore?.toFixed(2) ?? "—"}</td>
                    <td>{row.cheatingScore?.toFixed(2) ?? "—"}</td>
                    <td>{row.confidence?.toFixed(3) ?? "—"}</td>
                    <td>
                      {formatDistanceToNow(row.updatedAt, { addSuffix: true })}
                    </td>
                    <td>
                      <span className="muted">
                        {row.summary ? row.summary.slice(0, 120) : "—"}
                      </span>
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </section>
      </div>
    </div>
  );
}
