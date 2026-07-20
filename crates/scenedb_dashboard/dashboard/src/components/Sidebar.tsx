"use client";

import { usePathname } from "next/navigation";

const NAV_ITEMS = [
  { href: "/", label: "Overview", icon: DashboardIcon },
  { href: "/cells", label: "Cells", icon: CellsIcon },
  { href: "/gpu", label: "GPU", icon: GpuIcon },
  { href: "/pools", label: "Pools", icon: PoolsIcon },
  { href: "/queries", label: "Queries", icon: QueriesIcon },
  { href: "/schema", label: "Schema", icon: SchemaIcon },
];

export default function Sidebar({
  collapsed,
  onToggle,
}: {
  collapsed: boolean;
  onToggle: () => void;
}) {
  const pathname = usePathname();

  return (
    <aside
      className={`${
        collapsed ? "w-16" : "w-48"
      } flex-shrink-0 bg-github-sidebar flex flex-col min-h-screen transition-all duration-300 ease-in-out`}
    >
      <div
        className={`h-[68px] flex items-center shrink-0 overflow-hidden ${
          collapsed ? "gap-0 px-4" : "gap-3 px-6"
        }`}
      >
        <div className="w-7 h-7 rounded-lg bg-gradient-to-br from-github-accent to-github-blue flex items-center justify-center text-white text-[10px] font-bold shrink-0">
          SD
        </div>
        <div
          className={`flex flex-col leading-tight overflow-hidden transition-all duration-300 ${
            collapsed ? "max-w-0 opacity-0" : "max-w-48 opacity-100"
          }`}
        >
          <span className="text-sm font-semibold text-github-text whitespace-nowrap">SceneDB</span>
          <span className="text-[10px] text-github-text-muted whitespace-nowrap">Monitoring</span>
        </div>
      </div>

      <nav className="flex-1 px-3 py-4 space-y-1">
        {NAV_ITEMS.map((item) => {
          const isActive =
            item.href === "/"
              ? pathname === "/"
              : pathname.startsWith(item.href);
          return (
            <a
              key={item.href}
              href={item.href}
              className={`${isActive ? "nav-item-active" : "nav-item-inactive"} ${collapsed ? "justify-center gap-0 px-2" : ""}`}
              title={collapsed ? item.label : undefined}
            >
              <item.icon active={isActive} />
              <span
                className={`overflow-hidden whitespace-nowrap transition-all duration-300 ${
                  collapsed ? "max-w-0 opacity-0" : "max-w-48 opacity-100"
                }`}
              >
                {item.label}
              </span>
            </a>
          );
        })}
      </nav>

      <div className="px-3 pb-4 shrink-0 space-y-2">
        <div className="flex items-center gap-3 px-4 py-3 rounded-lg bg-github-bg/30 overflow-hidden">
          <div className="w-2 h-2 rounded-full bg-github-green shrink-0" />
          <span
            className={`text-xs text-github-text-secondary overflow-hidden whitespace-nowrap transition-all duration-300 ${
              collapsed ? "max-w-0 opacity-0" : "max-w-48 opacity-100"
            }`}
          >
            Connected
          </span>
        </div>
        <button
          onClick={onToggle}
          className="flex items-center justify-center w-full p-2 rounded-lg hover:bg-github-border-muted transition-colors text-github-text-muted hover:text-github-text"
          title={collapsed ? "Expand sidebar" : "Collapse sidebar"}
        >
          <svg
            className="w-6 h-6 shrink-0"
            fill="none"
            viewBox="0 0 24 24"
            stroke="currentColor"
            strokeWidth={2}
          >
            <path
              strokeLinecap="round"
              strokeLinejoin="round"
              d={collapsed ? "M9 18l6-6-6-6" : "M15 18l-6-6 6-6"}
            />
          </svg>
        </button>
      </div>
    </aside>
  );
}

function DashboardIcon({ active }: { active: boolean }) {
  return (
    <svg
      className={`w-6 h-6 ${active ? "text-github-text" : "text-github-text-muted"}`}
      fill="none"
      viewBox="0 0 24 24"
      stroke="currentColor"
      strokeWidth={2}
    >
      <rect x="3" y="3" width="7" height="7" rx="1" />
      <rect x="14" y="3" width="7" height="7" rx="1" />
      <rect x="3" y="14" width="7" height="7" rx="1" />
      <rect x="14" y="14" width="7" height="7" rx="1" />
    </svg>
  );
}

function CellsIcon({ active }: { active: boolean }) {
  return (
    <svg
      className={`w-6 h-6 ${active ? "text-github-text" : "text-github-text-muted"}`}
      fill="none"
      viewBox="0 0 24 24"
      stroke="currentColor"
      strokeWidth={2}
    >
      <path d="M3 9l9-7 9 7v11a2 2 0 01-2 2H5a2 2 0 01-2-2z" />
      <polyline points="9 22 9 12 15 12 15 22" />
    </svg>
  );
}

function GpuIcon({ active }: { active: boolean }) {
  return (
    <svg
      className={`w-6 h-6 ${active ? "text-github-text" : "text-github-text-muted"}`}
      fill="none"
      viewBox="0 0 24 24"
      stroke="currentColor"
      strokeWidth={2}
    >
      <rect x="2" y="3" width="20" height="14" rx="2" />
      <rect x="8" y="7" width="8" height="6" rx="1" />
      <circle cx="12" cy="10" r="1.5" fill="currentColor" />
    </svg>
  );
}

function PoolsIcon({ active }: { active: boolean }) {
  return (
    <svg
      className={`w-6 h-6 ${active ? "text-github-text" : "text-github-text-muted"}`}
      fill="none"
      viewBox="0 0 24 24"
      stroke="currentColor"
      strokeWidth={2}
    >
      <path d="M22 12h-4l-3 9L9 3l-3 9H2" />
    </svg>
  );
}

function QueriesIcon({ active }: { active: boolean }) {
  return (
    <svg
      className={`w-6 h-6 ${active ? "text-github-text" : "text-github-text-muted"}`}
      fill="none"
      viewBox="0 0 24 24"
      stroke="currentColor"
      strokeWidth={2}
    >
      <circle cx="11" cy="11" r="8" />
      <path d="M21 21l-4.35-4.35" />
    </svg>
  );
}

function SchemaIcon({ active }: { active: boolean }) {
  return (
    <svg
      className={`w-6 h-6 ${active ? "text-github-text" : "text-github-text-muted"}`}
      fill="none"
      viewBox="0 0 24 24"
      stroke="currentColor"
      strokeWidth={2}
    >
      <path d="M14 2H6a2 2 0 00-2 2v16a2 2 0 002 2h12a2 2 0 002-2V8z" />
      <polyline points="14 2 14 8 20 8" />
      <line x1="16" y1="13" x2="8" y2="13" />
      <line x1="16" y1="17" x2="8" y2="17" />
    </svg>
  );
}
