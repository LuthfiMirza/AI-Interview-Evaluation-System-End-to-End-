'use client';

import React, { useState } from "react";
import { Sidebar } from "./sidebar";
import { Header } from "./header";
import { SystemHealthModal } from "./system-health-modal";

type DashboardShellProps = {
  children: React.ReactNode;
  apiBaseUrl: string;
};

export function DashboardShell({ children, apiBaseUrl }: DashboardShellProps) {
  const [isCollapsed, setIsCollapsed] = useState(false);
  const [isHealthModalOpen, setIsHealthModalOpen] = useState(false);

  return (
    <div className="min-h-screen bg-[#F9FAFB] dark:bg-[#0A0F1D] flex">
      {/* Sidebar */}
      <Sidebar
        isCollapsed={isCollapsed}
        setIsCollapsed={setIsCollapsed}
        apiBaseUrl={apiBaseUrl}
        onOpenHealthModal={() => setIsHealthModalOpen(true)}
      />

      {/* Main Layout Area */}
      <div
        className={`flex-1 flex flex-col min-w-0 transition-all duration-300 ease-in-out ${
          isCollapsed ? "pl-20" : "pl-72"
        }`}
      >
        {/* Sticky Topbar */}
        <Header
          apiBaseUrl={apiBaseUrl}
          onOpenHealthModal={() => setIsHealthModalOpen(true)}
        />

        {/* Content Container */}
        <main className="flex-1 p-6 sm:p-8 lg:p-10 max-w-7xl w-full mx-auto">
          {children}
        </main>
      </div>

      {/* In-App Interactive System Health Modal */}
      <SystemHealthModal
        isOpen={isHealthModalOpen}
        onClose={() => setIsHealthModalOpen(false)}
        apiBaseUrl={apiBaseUrl}
      />
    </div>
  );
}
