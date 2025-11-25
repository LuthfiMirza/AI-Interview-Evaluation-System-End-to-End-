-- CreateTable
CREATE TABLE "Interview" (
    "id" TEXT NOT NULL,
    "candidateId" TEXT NOT NULL,
    "language" TEXT NOT NULL DEFAULT 'en',
    "status" TEXT NOT NULL DEFAULT 'processing',
    "createdAt" TIMESTAMP(3) NOT NULL DEFAULT CURRENT_TIMESTAMP,
    "updatedAt" TIMESTAMP(3) NOT NULL,
    "verbalScore" DOUBLE PRECISION,
    "nonVerbalScore" DOUBLE PRECISION,
    "cheatingScore" DOUBLE PRECISION,
    "finalScore" DOUBLE PRECISION,
    "confidence" DOUBLE PRECISION,
    "summary" TEXT,

    CONSTRAINT "Interview_pkey" PRIMARY KEY ("id")
);

-- CreateTable
CREATE TABLE "Transcript" (
    "id" TEXT NOT NULL,
    "interviewId" TEXT NOT NULL,
    "text" TEXT NOT NULL,
    "segments" JSONB,
    "createdAt" TIMESTAMP(3) NOT NULL DEFAULT CURRENT_TIMESTAMP,

    CONSTRAINT "Transcript_pkey" PRIMARY KEY ("id")
);

-- CreateTable
CREATE TABLE "NLPScore" (
    "id" TEXT NOT NULL,
    "interviewId" TEXT NOT NULL,
    "fluency" DOUBLE PRECISION NOT NULL,
    "relevance" DOUBLE PRECISION NOT NULL,
    "overall" DOUBLE PRECISION NOT NULL,
    "createdAt" TIMESTAMP(3) NOT NULL DEFAULT CURRENT_TIMESTAMP,

    CONSTRAINT "NLPScore_pkey" PRIMARY KEY ("id")
);

-- CreateTable
CREATE TABLE "VisionMetrics" (
    "id" TEXT NOT NULL,
    "interviewId" TEXT NOT NULL,
    "eyeContactRatio" DOUBLE PRECISION NOT NULL,
    "phoneDetected" BOOLEAN NOT NULL,
    "multiPerson" BOOLEAN NOT NULL,
    "cheatingScore" DOUBLE PRECISION NOT NULL,
    "createdAt" TIMESTAMP(3) NOT NULL DEFAULT CURRENT_TIMESTAMP,

    CONSTRAINT "VisionMetrics_pkey" PRIMARY KEY ("id")
);

-- CreateIndex
CREATE UNIQUE INDEX "Transcript_interviewId_key" ON "Transcript"("interviewId");

-- CreateIndex
CREATE UNIQUE INDEX "NLPScore_interviewId_key" ON "NLPScore"("interviewId");

-- CreateIndex
CREATE UNIQUE INDEX "VisionMetrics_interviewId_key" ON "VisionMetrics"("interviewId");

-- AddForeignKey
ALTER TABLE "Transcript" ADD CONSTRAINT "Transcript_interviewId_fkey" FOREIGN KEY ("interviewId") REFERENCES "Interview"("id") ON DELETE CASCADE ON UPDATE CASCADE;

-- AddForeignKey
ALTER TABLE "NLPScore" ADD CONSTRAINT "NLPScore_interviewId_fkey" FOREIGN KEY ("interviewId") REFERENCES "Interview"("id") ON DELETE CASCADE ON UPDATE CASCADE;

-- AddForeignKey
ALTER TABLE "VisionMetrics" ADD CONSTRAINT "VisionMetrics_interviewId_fkey" FOREIGN KEY ("interviewId") REFERENCES "Interview"("id") ON DELETE CASCADE ON UPDATE CASCADE;
