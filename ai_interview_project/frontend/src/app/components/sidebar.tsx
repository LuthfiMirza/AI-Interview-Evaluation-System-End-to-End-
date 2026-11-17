'use client';

import React from "react";
import Link from "next/link";
import { usePathname } from "next/navigation";
import {
  LayoutDashboard,
  Video,
  Users,
  Code2,
  Activity,
  FileCode,
  ShieldCheck,
  ChevronLeft,
  ChevronRight,
  Sparkles,
  ExternalLink,
} from "lucide-react";

type SidebarProps = {
  isCollapsed: boolean;
  setIsCollapsed: (collapsed: boolean) => void;
  apiBaseUrl: string;
  onOpenHealthModal?: () => void;
};

export function Sidebar({ isCollapsed, setIsCollapsed, apiBaseUrl, onOpenHealthModal }: SidebarProps) {
  const pathname = usePathname();

  const mainNav = [
    { label: "Dashboard Overview", href: "/", icon: LayoutDashboard, badge: "Live" },
    { label: "Video Assessment", href: "#upload-section", icon: Video },
    { label: "Candidate Directory", href: "#candidates-section", icon: Users },
  ];

  const evalNav = [
    { label: "React & Next.js Rubric", href: "#rubric-section", icon: Code2, badge: "New" },
    { label: "Speech & NLP Metrics", href: "#analytics-section", icon: Activity },
  ];

  const swaggerUrl = `${apiBaseUrl.replace('/api', '')}/docs`;

  return (
    <aside
      className={`fixed inset-y-0 left-0 z-40 flex flex-col bg-white dark:bg-[#101828] border-r border-gray-200 dark:border-gray-800 transition-all duration-300 ease-in-out ${
        isCollapsed ? "w-20" : "w-72"
      }`}
    >
      {/* Brand Header */}
      <div className="flex items-center justify-between h-20 px-5 border-b border-gray-100 dark:border-gray-800/80">
        <Link href="/" className="flex items-center gap-3 overflow-hidden">
          <div className="flex items-center justify-center w-10 h-10 rounded-xl bg-gradient-to-tr from-brand-600 to-indigo-500 text-white shadow-md shadow-brand-500/20 shrink-0">
            <Sparkles className="w-5 h-5" />
          </div>
          {!isCollapsed && (
            <div className="flex flex-col">
              <span className="font-bold text-base tracking-tight text-gray-900 dark:text-white leading-tight">
                AI Interview
              </span>
              <span className="text-[11px] font-medium text-brand-600 dark:text-brand-400">
                Evaluation System
              </span>
            </div>
          )}
        </Link>
        <button
          onClick={() => setIsCollapsed(!isCollapsed)}
          className="p-1.5 rounded-lg text-gray-400 hover:text-gray-600 hover:bg-gray-100 dark:hover:bg-gray-800 transition-colors"
          title={isCollapsed ? "Expand Sidebar" : "Collapse Sidebar"}
        >
          {isCollapsed ? <ChevronRight className="w-4 h-4" /> : <ChevronLeft className="w-4 h-4" />}
        </button>
      </div>

      {/* Navigation Groups */}
      <div className="flex-1 overflow-y-auto px-4 py-5 space-y-6">
        {/* Main Section */}
        <div>
          {!isCollapsed && (
            <p className="px-3 mb-2 text-[11px] font-semibold tracking-wider text-gray-400 uppercase">
              Main Menu
            </p>
          )}
          <nav className="space-y-1">
            {mainNav.map((item) => {
              const Icon = item.icon;
              const isActive = pathname === item.href || (item.href === "/" && pathname === "/");
              return (
                <Link
                  key={item.label}
                  href={item.href}
                  className={`flex items-center gap-3 px-3.5 py-2.5 rounded-xl text-sm font-medium transition-all ${
                    isActive
                      ? "bg-brand-50 text-brand-600 dark:bg-brand-500/10 dark:text-brand-400"
                      : "text-gray-600 hover:bg-gray-100 dark:text-gray-300 dark:hover:bg-gray-800/60"
                  }`}
                  title={isCollapsed ? item.label : undefined}
                >
                  <Icon className="w-5 h-5 shrink-0" />
                  {!isCollapsed && (
                    <span className="flex-1 truncate">{item.label}</span>
                  )}
                  {!isCollapsed && item.badge && (
                    <span className="px-2 py-0.5 text-[10px] font-semibold rounded-full bg-emerald-50 text-emerald-600 dark:bg-emerald-500/15 dark:text-emerald-400">
                      {item.badge}
                    </span>
                  )}
                </Link>
              );
            })}
          </nav>
        </div>

        {/* Evaluation & Rubric Section */}
        <div>
          {!isCollapsed && (
            <p className="px-3 mb-2 text-[11px] font-semibold tracking-wider text-gray-400 uppercase">
              Evaluation & Engine
            </p>
          )}
          <nav className="space-y-1">
            {evalNav.map((item) => {
              const Icon = item.icon;
              return (
                <Link
                  key={item.label}
                  href={item.href}
                  className="flex items-center gap-3 px-3.5 py-2.5 rounded-xl text-sm font-medium text-gray-600 hover:bg-gray-100 dark:text-gray-300 dark:hover:bg-gray-800/60 transition-all"
                  title={isCollapsed ? item.label : undefined}
                >
                  <Icon className="w-5 h-5 shrink-0" />
                  {!isCollapsed && (
                    <span className="flex-1 truncate">{item.label}</span>
                  )}
                  {!isCollapsed && item.badge && (
                    <span className="px-2 py-0.5 text-[10px] font-semibold rounded-full bg-brand-50 text-brand-600 dark:bg-brand-500/15 dark:text-brand-400">
                      {item.badge}
                    </span>
                  )}
                </Link>
              );
            })}
          </nav>
        </div>

        {/* Backend & Diagnostics Section */}
        <div>
          {!isCollapsed && (
            <p className="px-3 mb-2 text-[11px] font-semibold tracking-wider text-gray-400 uppercase">
              Backend & Status
            </p>
          )}
          <nav className="space-y-1">
            {/* Interactive Health Check Modal Trigger */}
            <button
              type="button"
              onClick={onOpenHealthModal}
              className="w-full flex items-center gap-3 px-3.5 py-2.5 rounded-xl text-sm font-medium text-gray-600 hover:bg-gray-100 dark:text-gray-300 dark:hover:bg-gray-800/60 transition-all text-left"
              title={isCollapsed ? "Status Sistem & Diagnostik" : undefined}
            >
              <ShieldCheck className="w-5 h-5 shrink-0 text-emerald-500" />
              {!isCollapsed && (
                <>
                  <span className="flex-1 truncate">Status Sistem & Diagnostik</span>
                  <span className="w-2 h-2 rounded-full bg-emerald-500" />
                </>
              )}
            </button>

            {/* Swagger Docs External Link */}
            <a
              href={swaggerUrl}
              target="_blank"
              rel="noopener noreferrer"
              className="flex items-center gap-3 px-3.5 py-2.5 rounded-xl text-sm font-medium text-gray-600 hover:bg-gray-100 dark:text-gray-300 dark:hover:bg-gray-800/60 transition-all"
              title={isCollapsed ? "FastAPI Swagger Docs" : undefined}
            >
              <FileCode className="w-5 h-5 shrink-0 text-brand-500" />
              {!isCollapsed && (
                <>
                  <span className="flex-1 truncate">FastAPI Docs (Swagger)</span>
                  <ExternalLink className="w-3.5 h-3.5 text-gray-400" />
                </>
              )}
            </a>
          </nav>
        </div>
      </div>

      {/* Recruiter User Footer Profile */}
      <div className="p-4 border-t border-gray-100 dark:border-gray-800/80">
        <div className="flex items-center gap-3 p-2 rounded-xl bg-gray-50 dark:bg-gray-900/50">
          <div className="relative flex items-center justify-center w-9 h-9 rounded-full bg-brand-100 text-brand-700 font-bold text-sm shrink-0">
            HR
            <span className="absolute bottom-0 right-0 w-2.5 h-2.5 rounded-full bg-emerald-500 ring-2 ring-white dark:ring-[#101828]" />
          </div>
          {!isCollapsed && (
            <div className="flex flex-col min-w-0">
              <span className="text-sm font-semibold text-gray-900 dark:text-white truncate">
                Tech Recruiter
              </span>
              <span className="text-xs text-gray-400 truncate">
                AI Assessment Active
              </span>
            </div>
          )}
        </div>
      </div>
    </aside>
  );
}
