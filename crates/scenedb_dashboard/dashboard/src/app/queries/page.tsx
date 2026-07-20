"use client";

import { useState } from "react";
import { usePoll, fetchStats, type StatsSnapshot } from "@/lib/api";

interface QueryLogEntry {
  timestamp: string;
  type: string;
  cell: number;
  duration_us: number;
  rows: number;
}

export default function QueriesPage() {
  const [stats, setStats] = useState<StatsSnapshot | null>(null);
  const [tab, setTab] = useState<"recent" | "frequent">("recent");

  usePoll(async () => setStats(await fetchStats()), []);

  const recentQueries: QueryLogEntry[] = [
    { timestamp: new Date().toISOString(), type: "AABB", cell: 0, duration_us: 142, rows: 64 },
    { timestamp: new Date(Date.now() - 1000).toISOString(), type: "Frustum", cell: 1, duration_us: 89, rows: 32 },
    { timestamp: new Date(Date.now() - 2000).toISOString(), type: "ECS Query", cell: 0, duration_us: 210, rows: 128 },
    { timestamp: new Date(Date.now() - 3000).toISOString(), type: "AABB", cell: 2, duration_us: 55, rows: 8 },
    { timestamp: new Date(Date.now() - 4000).toISOString(), type: "Frustum", cell: 1, duration_us: 176, rows: 96 },
    { timestamp: new Date(Date.now() - 5000).toISOString(), type: "ECS Query", cell: 0, duration_us: 98, rows: 16 },
    { timestamp: new Date(Date.now() - 6000).toISOString(), type: "AABB", cell: 3, duration_us: 320, rows: 256 },
    { timestamp: new Date(Date.now() - 7000).toISOString(), type: "Frustum", cell: 0, duration_us: 44, rows: 4 },
    { timestamp: new Date(Date.now() - 8000).toISOString(), type: "AABB", cell: 1, duration_us: 267, rows: 192 },
    { timestamp: new Date(Date.now() - 9000).toISOString(), type: "ECS Query", cell: 2, duration_us: 133, rows: 48 },
  ];

  const frequentQueries: { query: string; count: number; avg_us: number }[] = [
    { query: "AABB cell=0 col=0", count: 1247, avg_us: 138 },
    { query: "Frustum cell=1 col=0", count: 892, avg_us: 94 },
    { query: "ECS Query (Transform, Position)", count: 654, avg_us: 187 },
    { query: "AABB cell=3 col=0", count: 423, avg_us: 301 },
    { query: "Frustum cell=0 col=0", count: 312, avg_us: 52 },
  ];

  return (
    <div className="space-y-6">
      <div>
        <h1 className="text-xl font-semibold">Queries</h1>
        <p className="text-sm text-github-text-secondary mt-1">Query log &mdash; 15 fps &mdash; requires query logging feature on SceneDB</p>
      </div>

      <div className="card">
        <div className="flex items-center gap-2 mb-4">
          <button onClick={() => setTab("recent")}
            className={`px-4 py-1.5 text-sm rounded-lg transition-colors ${tab === "recent" ? "bg-github-accent/10 text-github-accent font-medium" : "text-github-text-secondary hover:text-github-text"}`}>
            Recent 100
          </button>
          <button onClick={() => setTab("frequent")}
            className={`px-4 py-1.5 text-sm rounded-lg transition-colors ${tab === "frequent" ? "bg-github-accent/10 text-github-accent font-medium" : "text-github-text-secondary hover:text-github-text"}`}>
            Most Common 100
          </button>
        </div>

        {tab === "recent" ? (
          <div className="overflow-x-auto">
            <table className="w-full">
              <thead><tr>
                <th className="table-header">Time</th>
                <th className="table-header">Type</th>
                <th className="table-header">Cell</th>
                <th className="table-header">Duration</th>
                <th className="table-header">Rows</th>
              </tr></thead>
              <tbody>
                {recentQueries.map((q, i) => (
                  <tr key={i} className="hover:bg-github-border-muted/30">
                    <td className="table-cell text-xs text-github-text-muted font-mono">{new Date(q.timestamp).toLocaleTimeString()}</td>
                    <td className="table-cell">
                      <span className={`badge ${q.type === "AABB" ? "badge-blue" : q.type === "Frustum" ? "badge-yellow" : "badge-green"}`}>{q.type}</span>
                    </td>
                    <td className="table-cell font-mono text-github-accent">#{q.cell}</td>
                    <td className={`table-cell font-mono ${q.duration_us > 200 ? "text-github-red" : q.duration_us > 100 ? "text-github-yellow" : "text-github-green"}`}>{q.duration_us}µs</td>
                    <td className="table-cell font-mono">{q.rows}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        ) : (
          <div className="overflow-x-auto">
            <table className="w-full">
              <thead><tr>
                <th className="table-header">#</th>
                <th className="table-header">Query Pattern</th>
                <th className="table-header">Count</th>
                <th className="table-header">Avg Duration</th>
                <th className="table-header">Total Time</th>
              </tr></thead>
              <tbody>
                {frequentQueries.map((q, i) => (
                  <tr key={i} className="hover:bg-github-border-muted/30">
                    <td className="table-cell text-github-text-muted">{i + 1}</td>
                    <td className="table-cell font-mono text-xs text-github-accent">{q.query}</td>
                    <td className="table-cell font-mono">{q.count.toLocaleString()}</td>
                    <td className="table-cell font-mono">{q.avg_us}µs</td>
                    <td className="table-cell font-mono">{(q.count * q.avg_us / 1000).toFixed(0)}ms</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        )}

        <div className="mt-4 pt-4 border-t border-github-border-muted">
          <div className="flex items-center gap-2 text-xs text-github-text-muted">
            <svg className="w-4 h-4 text-github-yellow shrink-0" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2}>
              <path d="M12 9v2m0 4h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z" />
            </svg>
            Query data shown is sample/demo. SceneDB does not yet log queries — requires a query logging ring buffer in the Rust telemetry layer.
          </div>
        </div>
      </div>
    </div>
  );
}
