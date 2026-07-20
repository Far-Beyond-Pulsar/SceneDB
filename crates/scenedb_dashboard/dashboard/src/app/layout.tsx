import type { Metadata } from "next";
import "./globals.css";
import Sidebar from "@/components/Sidebar";

export const metadata: Metadata = {
  title: "SceneDB Dashboard",
  description: "Real-time SceneDB monitoring dashboard",
};

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="en" className="dark">
      <body>
        <div className="flex min-h-screen">
          <Sidebar />
          <div className="flex-1 flex flex-col min-w-0">
            <header className="h-[68px] bg-github-sidebar flex items-center px-8 gap-4 border-b border-github-border-muted">
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
            <main className="flex-1 overflow-auto">
              <div className="p-8">{children}</div>
            </main>
          </div>
        </div>
      </body>
    </html>
  );
}
