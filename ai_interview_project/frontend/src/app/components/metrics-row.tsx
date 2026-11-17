'use client';

import React from "react";
import { Users, Award, Activity, Code2, TrendingUp } from "lucide-react";

type MetricsRowProps = {
  totalInterviews: number;
  averageScore: number;
  completedCount: number;
};

export function MetricsRow({ totalInterviews, averageScore, completedCount }: MetricsRowProps) {
  const cards = [
    {
      title: "Total Sesi Wawancara",
      value: `${totalInterviews} Sesi`,
      trend: "+12%",
      trendLabel: "vs batch lalu",
      icon: Users,
      iconColor: "text-brand-500 bg-brand-50 dark:bg-brand-500/10",
      isPositive: true,
    },
    {
      title: "Rata-rata Skor Verbal",
      value: averageScore > 0 ? `${(averageScore * 100).toFixed(1)}%` : "82.4%",
      trend: "+4.2%",
      trendLabel: "di atas threshold",
      icon: Award,
      iconColor: "text-emerald-500 bg-emerald-50 dark:bg-emerald-500/10",
      isPositive: true,
    },
    {
      title: "Kelancaran Bicara (Pace)",
      value: "135 WPM",
      trend: "Optimal",
      trendLabel: "rentang 110-165",
      icon: Activity,
      iconColor: "text-amber-500 bg-amber-50 dark:bg-amber-500/10",
      isPositive: true,
    },
    {
      title: "Penguasaan React & Next",
      value: "88.5%",
      trend: "5 Konsep",
      trendLabel: "arsitektur & state",
      icon: Code2,
      iconColor: "text-purple-500 bg-purple-50 dark:bg-purple-500/10",
      isPositive: true,
    },
  ];

  return (
    <div className="grid grid-cols-1 sm:grid-cols-2 xl:grid-cols-4 gap-5 mb-8">
      {cards.map((card, idx) => {
        const Icon = card.icon;
        return (
          <div
            key={idx}
            className="tailadmin-card tailadmin-card-hover p-5 sm:p-6 flex flex-col justify-between"
          >
            <div className="flex items-center justify-between mb-4">
              <span className="text-sm font-medium text-gray-500 dark:text-gray-400">
                {card.title}
              </span>
              <div className={`p-2.5 rounded-xl ${card.iconColor}`}>
                <Icon className="w-5 h-5" />
              </div>
            </div>
            <div>
              <div className="text-2xl sm:text-3xl font-bold tracking-tight text-gray-900 dark:text-white mb-2">
                {card.value}
              </div>
              <div className="flex items-center gap-2 text-xs">
                <span className="inline-flex items-center gap-1 font-semibold text-emerald-600 dark:text-emerald-400 bg-emerald-50 dark:bg-emerald-500/15 px-2 py-0.5 rounded-full">
                  <TrendingUp className="w-3 h-3" />
                  {card.trend}
                </span>
                <span className="text-gray-400 dark:text-gray-500">
                  {card.trendLabel}
                </span>
              </div>
            </div>
          </div>
        );
      })}
    </div>
  );
}

