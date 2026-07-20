"use client";

import { useState } from "react";
import QueryShell from "@/components/QueryShell";
import { usePoll } from "@/lib/api";

interface QueryLogEntry {
  timestamp_ms: number;
  query_type: string;
  cell_id: number;
  duration_ns: number;
  rows_returned: number;
  total_rows: number;
  args: string;
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

async function fetchSlow(): Promise<QueryLogEntry[]> {
  const res = await fetch("/api/queries/slow");
  return res.json();
}

export default function QueriesPage() {
  const [tab, setTab] = useState<"shell" | "recent" | "frequent" | "slow">("shell");
  const [recent, setRecent] = useState<QueryLogEntry[]>([]);
  const [frequent, setFrequent] = useState<FrequentQuery[]>([]);
  const [slow, setSlow] = useState<QueryLogEntry[]>([]);

  usePoll(async () => {
    if (tab === "recent") setRecent(await fetchRecent());
    else if (tab === "frequent") setFrequent(await fetchFrequent());
    else if (tab === "slow") setSlow(await fetchSlow());
  }, [tab]);

  return (
    <div className="space-y-6">
      <div className="flex items-center gap-4">
        <h1 className="text-xl font-semibold">Queries</h1>
        <div className="flex items-center gap-1 bg-github-card rounded-lg p-0.5 border border-github-border-muted">
          {(["shell", "recent", "frequent", "slow"] as const).map((t) => (
            <button key={t} onClick={() => setTab(t)}
              className={`px-3 py-1.5 text-sm rounded-md transition-colors capitalize ${
                tab === t ? "bg-github-accent/10 text-github-accent font-medium" : "text-github-text-secondary hover:text-github-text"
              }`}
            >{t === "shell" ? "Shell" : t === "recent" ? `Recent (${recent.length})` : t === "frequent" ? `Frequent (${frequent.length})` : `Slow (${slow.length})`}</button>
          ))}
        </div>
      </div>

      {tab === "shell" ? (
        <QueryShell />
      ) : tab === "recent" ? (
        <div className="card">
          {recent.length === 0 ? (
            <div className="text-center py-12">
              <p className="text-sm text-github-text-muted">No queries logged yet</p>
              <p className="text-xs text-github-text-muted mt-2">Run spatial queries against SpatialCell to populate</p>
            </div>
          ) : (
            <div className="overflow-x-auto">
              <table className="w-full">
                <thead><tr>
                  <th className="table-header">Time</th>
                  <th className="table-header">Type</th>
                  <th className="table-header">Args</th>
                  <th className="table-header">Cell</th>
                  <th className="table-header">Duration</th>
                  <th className="table-header">Rows</th>
                </tr></thead>
                <tbody>
                  {recent.slice(0, 100).map((q, i) => (
                    <tr key={i} className="hover:bg-github-border-muted/30">
                      <td className="table-cell text-xs font-mono text-github-text-muted">{new Date(q.timestamp_ms).toLocaleTimeString()}</td>
                      <td className="table-cell"><span className={`badge ${q.query_type === "AABB" ? "badge-blue" : "badge-yellow"}`}>{q.query_type}</span></td>
                      <td className="table-cell text-xs font-mono text-github-text-secondary max-w-[200px] truncate" title={q.args}>{q.args}</td>
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
          )}
        </div>
      ) : tab === "frequent" ? (
        <div className="card">
          {frequent.length === 0 ? (
            <div className="text-center py-12"><p className="text-sm text-github-text-muted">No frequent data yet</p></div>
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
                      <td className="table-cell"><span className={`badge ${q.query_type === "AABB" ? "badge-blue" : "badge-yellow"}`}>{q.query_type}</span></td>
                      <td className="table-cell font-mono text-github-accent">#{q.cell_id}</td>
                      <td className="table-cell font-mono">{q.count.toLocaleString()}</td>
                      <td className="table-cell font-mono">{(q.avg_duration_ns / 1000).toFixed(0)}µs</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          )}
        </div>
      ) : (
        <div className="card">
          {slow.length === 0 ? (
            <div className="text-center py-12">
              <p className="text-sm text-github-text-muted">No slow queries</p>
              <p className="text-xs text-github-text-muted mt-2">Queries exceeding 200µs appear here</p>
            </div>
          ) : (
            <div className="overflow-x-auto">
              <table className="w-full">
                <thead><tr>
                  <th className="table-header">Time</th>
                  <th className="table-header">Type</th>
                  <th className="table-header">Args</th>
                  <th className="table-header">Cell</th>
                  <th className="table-header">Duration</th>
                  <th className="table-header">Rows</th>
                </tr></thead>
                <tbody>
                  {slow.slice(0, 100).map((q, i) => (
                    <tr key={i} className="hover:bg-github-border-muted/30">
                      <td className="table-cell text-xs font-mono text-github-text-muted">{new Date(q.timestamp_ms).toLocaleTimeString()}</td>
                      <td className="table-cell"><span className={`badge ${q.query_type === "AABB" ? "badge-blue" : "badge-yellow"}`}>{q.query_type}</span></td>
                      <td className="table-cell text-xs font-mono text-github-text-secondary max-w-[200px] truncate" title={q.args}>{q.args}</td>
                      <td className="table-cell font-mono text-github-accent">#{q.cell_id}</td>
                      <td className="table-cell font-mono text-github-red font-semibold">{(q.duration_ns / 1000).toFixed(0)}µs</td>
                      <td className="table-cell font-mono text-github-text-secondary">{q.rows_returned} / {q.total_rows}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          )}
        </div>
      )}
    </div>
  );
}
