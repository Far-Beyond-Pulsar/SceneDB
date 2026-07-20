"use client";

import { useState } from "react";
import { usePoll } from "@/lib/api";

interface QueryLogEntry {
  timestamp_ms: number;
  query_type: string;
  cell_id: number;
  duration_ns: number;
  rows_returned: number;
  total_rows: number;
}

interface FrequentQuery {
  query_type: string;
  cell_id: number;
  count: number;
  avg_duration_ns: number;
}

async function fetchRecent(): Promise<QueryLogEntry[]> {
  const res = await fetch("/api/queries/recent");
  return res.json();
}

async function fetchFrequent(): Promise<FrequentQuery[]> {
  const res = await fetch("/api/queries/frequent");
  return res.json();
}

export default function QueriesPage() {
  const [recent, setRecent] = useState<QueryLogEntry[]>([]);
  const [frequent, setFrequent] = useState<FrequentQuery[]>([]);
  const [tab, setTab] = useState<"recent" | "frequent">("recent");
  const [connected, setConnected] = useState(false);

  usePoll(async () => {
    const [r, f] = await Promise.all([fetchRecent(), fetchFrequent()]);
    setRecent(r); setFrequent(f); setConnected(true);
  }, []);

  return (
    <div className="space-y-6">
      <div>
        <h1 className="text-xl font-semibold">Queries</h1>
        <p className="text-sm text-github-text-secondary mt-1">
          {recent.length} recent entries &middot; {frequent.length} frequent patterns &middot; 15 fps
        </p>
      </div>

      <div className="card">
        <div className="flex items-center gap-2 mb-4">
          <button onClick={() => setTab("recent")}
            className={`px-4 py-1.5 text-sm rounded-lg transition-colors ${tab === "recent" ? "bg-github-accent/10 text-github-accent font-medium" : "text-github-text-secondary hover:text-github-text"}`}>
            Recent {recent.length > 0 ? `(${recent.length})` : ""}
          </button>
          <button onClick={() => setTab("frequent")}
            className={`px-4 py-1.5 text-sm rounded-lg transition-colors ${tab === "frequent" ? "bg-github-accent/10 text-github-accent font-medium" : "text-github-text-secondary hover:text-github-text"}`}>
            Most Common {frequent.length > 0 ? `(${frequent.length})` : ""}
          </button>
        </div>

        {tab === "recent" ? (
          recent.length === 0 ? (
            <div className="text-center py-12">
              <p className="text-sm text-github-text-muted">No queries logged yet</p>
              <p className="text-xs text-github-text-muted mt-2">Run spatial queries (AABB/Frustum) against SpatialCell to populate the query log</p>
            </div>
          ) : (
            <div className="overflow-x-auto">
              <table className="w-full">
                <thead><tr>
                  <th className="table-header">Time</th>
                  <th className="table-header">Type</th>
                  <th className="table-header">Cell</th>
                  <th className="table-header">Duration</th>
                  <th className="table-header">Rows / Total</th>
                </tr></thead>
                <tbody>
                  {recent.slice(0, 100).map((q, i) => (
                    <tr key={i} className="hover:bg-github-border-muted/30">
                      <td className="table-cell text-xs text-github-text-muted font-mono">
                        {new Date(q.timestamp_ms).toLocaleTimeString()}
                      </td>
                      <td className="table-cell">
                        <span className={`badge ${q.query_type === "AABB" ? "badge-blue" : "badge-yellow"}`}>{q.query_type}</span>
                      </td>
                      <td className="table-cell font-mono text-github-accent">#{q.cell_id}</td>
                      <td className={`table-cell font-mono ${q.duration_ns > 200_000 ? "text-github-red" : q.duration_ns > 100_000 ? "text-github-yellow" : "text-github-green"}`}>
                        {(q.duration_ns / 1000).toFixed(0)}µs
                      </td>
                      <td className="table-cell font-mono text-github-text-secondary">{q.rows_returned} / {q.total_rows}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          )
        ) : (
          frequent.length === 0 ? (
            <div className="text-center py-12">
              <p className="text-sm text-github-text-muted">No frequent query data yet</p>
            </div>
          ) : (
            <div className="overflow-x-auto">
              <table className="w-full">
                <thead><tr>
                  <th className="table-header">#</th>
                  <th className="table-header">Type</th>
                  <th className="table-header">Cell</th>
                  <th className="table-header">Count</th>
                  <th className="table-header">Avg Duration</th>
                </tr></thead>
                <tbody>
                  {frequent.map((q, i) => (
                    <tr key={i} className="hover:bg-github-border-muted/30">
                      <td className="table-cell text-github-text-muted">{i + 1}</td>
                      <td className="table-cell">
                        <span className={`badge ${q.query_type === "AABB" ? "badge-blue" : "badge-yellow"}`}>{q.query_type}</span>
                      </td>
                      <td className="table-cell font-mono text-github-accent">#{q.cell_id}</td>
                      <td className="table-cell font-mono">{q.count.toLocaleString()}</td>
                      <td className="table-cell font-mono">{(q.avg_duration_ns / 1000).toFixed(0)}µs</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          )
        )}

        {!connected && recent.length === 0 && (
          <div className="mt-4 pt-4 border-t border-github-border-muted">
            <div className="flex items-center gap-2 text-xs text-github-text-muted">
              <svg className="w-4 h-4 text-github-yellow shrink-0" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2}>
                <path d="M12 9v2m0 4h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z" />
              </svg>
              Query logging active. Run spatial queries (AABB/Frustum) to populate this page.
            </div>
          </div>
        )}
      </div>
    </div>
  );
}
