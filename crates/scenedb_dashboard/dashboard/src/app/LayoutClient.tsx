"use client";

import { useState } from "react";
import Sidebar from "@/components/Sidebar";

export default function LayoutClient({ children }: { children: React.ReactNode }) {
  const [sidebarCollapsed, setSidebarCollapsed] = useState(false);

  return (
    <div className="flex min-h-screen">
      <Sidebar
        collapsed={sidebarCollapsed}
        onToggle={() => setSidebarCollapsed(!sidebarCollapsed)}
      />
      <div className="flex-1 flex flex-col min-w-0 bg-github-sidebar">
        <header className="h-[68px] flex items-center px-8 gap-4 shrink-0">
          <div className="flex-1" />
          <div className="flex items-center gap-3 text-xs text-github-text-muted">
            <span className="flex items-center gap-1.5">
              <span className="w-1.5 h-1.5 rounded-full bg-github-green" />
              Live
            </span>
            <span className="text-github-border">|</span>
            <span>v0.1.0</span>
          </div>
        </header>
        <main className="flex-1 overflow-auto rounded-tl-[24px] border-t border-l border-github-border-muted bg-github-bg">
          <div className="p-8">{children}</div>
        </main>
      </div>
    </div>
  );
}
