import { format } from "date-fns";

type ReportCardProps = {
  interview: {
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
  } | null;
};

const formatNumber = (value: number | null, digits = 2, fallback = "—") =>
  typeof value === "number" ? value.toFixed(digits) : fallback;

export function ReportCard({ interview }: ReportCardProps) {
  if (!interview) {
    return (
      <div className="muted">
        No report yet. Upload a video and click <strong>Process</strong> to see
        transcription, metrics, and export-ready notes.
      </div>
    );
  }

  const progress = (interview.finalScore ?? 0) * 100;
  const confidence = Math.min((interview.confidence ?? 0) * 100, 100);

  return (
    <div className="report-card">
      <header>
        <h4 style={{ margin: 0, fontSize: "1.2rem", fontWeight: 600 }}>
          {interview.candidateId}
        </h4>
        <p className="muted" style={{ fontSize: "0.85rem" }}>
          Updated {format(interview.updatedAt, "dd MMM yyyy • HH:mm")}
        </p>
      </header>

      <div className="report-grid">
        <Metric label="Final Score" value={formatNumber(interview.finalScore)} />
        <Metric label="Status" value={interview.status} />
        <Metric
          label="Verbal Readout"
          value={formatNumber(interview.verbalScore)}
        />
        <Metric
          label="Non-Verbal"
          value={formatNumber(interview.nonVerbalScore)}
        />
        <Metric
          label="Cheating Risk"
          value={formatNumber(interview.cheatingScore)}
        />
        <Metric
          label="Confidence"
          value={`${confidence.toFixed(1)}%`}
        />
      </div>

      <div className="progress-shell">
        <header>
          <span>Score Gauge</span>
          <span>{progress.toFixed(1)}%</span>
        </header>
        <div className="progress-bar">
          <div
            className="progress-value"
            style={{ width: `${Math.min(progress, 100)}%` }}
          />
        </div>
      </div>

      <section>
        <h5 style={{ marginBottom: 6 }}>Interview Notes</h5>
        <p className="muted" style={{ lineHeight: 1.6 }}>
          {interview.summary ?? "Summary not captured for this session."}
        </p>
      </section>

      <section>
        <h5 style={{ marginBottom: 6 }}>HR Recommendation Template</h5>
        <ul style={{ margin: 0, paddingLeft: 18, color: "var(--muted)" }}>
          <li>
            <strong>Strengths:</strong> Highlight key behavioural / technical
            cues observed here.
          </li>
          <li>
            <strong>Risks:</strong> Review cheating score & non-verbal anomalies
            for manual verification.
          </li>
          <li>
            <strong>Next steps:</strong> Plan follow-up rounds or offer issuance
            based on the overall score + confidence.
          </li>
        </ul>
      </section>
    </div>
  );
}

type MetricProps = {
  label: string;
  value: string;
};

function Metric({ label, value }: MetricProps) {
  return (
    <div className="report-metric">
      <span>{label}</span>
      <strong>{value}</strong>
    </div>
  );
}
