'use client';

import { FormEvent, useMemo, useState } from "react";

type UploadPanelProps = {
  apiBaseUrl: string;
};

type UploadStatus = "idle" | "uploading" | "success" | "error";

export function UploadPanel({ apiBaseUrl }: UploadPanelProps) {
  const [candidateId, setCandidateId] = useState("");
  const [expectedAnswer, setExpectedAnswer] = useState("");
  const [file, setFile] = useState<File | null>(null);
  const [status, setStatus] = useState<UploadStatus>("idle");
  const [message, setMessage] = useState<string | null>(null);
  const [interviewId, setInterviewId] = useState<string | null>(null);

  const disabled = useMemo(() => status === "uploading", [status]);

  async function handleSubmit(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();
    if (!file) {
      setStatus("error");
      setMessage("Please attach an interview video file before submitting.");
      return;
    }

    try {
      setStatus("uploading");
      setMessage("Uploading video and queuing background evaluation…");

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
        throw new Error(`Upload failed with status ${response.status}`);
      }

      const data = (await response.json()) as { interview_id: string };
      setInterviewId(data.interview_id);
      setStatus("success");
      setMessage(
        "Upload succeeded. The background worker is analysing the video. Poll the result endpoint to refresh the dashboard.",
      );
      setFile(null);
      (event.target as HTMLFormElement).reset();
    } catch (error) {
      console.error(error);
      setStatus("error");
      setMessage(
        error instanceof Error ? error.message : "Unexpected upload error.",
      );
    }
  }

  return (
    <>
      <form className="upload-form" onSubmit={handleSubmit}>
        <div className="field">
          <label>Candidate ID (optional)</label>
          <input
            type="text"
            placeholder="CAND-123"
            className="input"
            onChange={(event) => setCandidateId(event.target.value)}
            disabled={disabled}
          />
        </div>

        <div className="field">
          <label>Expected Answer / Prompt (optional)</label>
          <textarea
            className="textarea"
            placeholder="Outline the ideal response so relevance scoring has context."
            onChange={(event) => setExpectedAnswer(event.target.value)}
            disabled={disabled}
          />
        </div>

        <div className="field">
          <label>Upload Video</label>
          <label className="upload-dropzone">
            <input
              type="file"
              accept="video/mp4,video/webm,video/quicktime"
              style={{ display: "none" }}
              onChange={(event) => setFile(event.target.files?.[0] ?? null)}
              disabled={disabled}
            />
            <strong>Drag & drop or click to select a file</strong>
            <span className="muted">
              MP4 · MOV · MKV (recommended ≤1 GB per recording)
            </span>
          </label>
        </div>

        <button type="submit" className="button" disabled={disabled}>
          {status === "uploading" ? "Processing…" : "Process Interview"}
        </button>
      </form>

      {message && (
        <div
          className={`alert ${
            status === "error" ? "alert-error" : "alert-success"
          }`}
        >
          {message}
        </div>
      )}

      {interviewId && (
        <div className="muted" style={{ fontSize: "0.85rem" }}>
          Latest reference ID: <strong>{interviewId}</strong>. Poll{" "}
          <code>{apiBaseUrl}/interviews/result/{interviewId}</code> to retrieve
          the final report.
        </div>
      )}
    </>
  );
}
