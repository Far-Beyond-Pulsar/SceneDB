"use client";

import { useState } from "react";
import StatCard from "@/components/StatCard";
import { usePoll, fetchStats, fetchCells, fetchPools, type StatsSnapshot, type CellSnapshot, type PoolSnapshot } from "@/lib/api";

export default function Overview() {
  const [stats, setStats] = useState<StatsSnapshot | null>(null);
  const [cells, setCells] = useState<CellSnapshot[]>([]);
  const [pools, setPools] = useState<PoolSnapshot | null>(null);
  const [connected, setConnected] = useState(true);

  usePoll(async () => {
    const s = await fetchStats();
    setStats(s); setConnected(true);
    try {
      const [c, p] = await Promise.all([fetchCells(), fetchPools()]);
      setCells(c); setPools(p);
    } catch { /* detail fetch best-effort */ }
  }, []);

  const alive = cells.filter((c) => c.rows_in_use > 0).length;
  const totalRows = cells.reduce((s, c) => s + c.rows_in_use, 0);
  const poolFree = pools
    ? Math.round(
        pools.row.reduce((s, p) => s + p.free, 0) /
          Math.max(pools.row.reduce((s, p) => s + p.total, 0), 1) * 100,
      )
    : 0;

  return (
    <div className="space-y-8">
      <div className="flex items-center gap-3">
        <div>
          <h1 className="text-xl font-semibold">Dashboard</h1>
          <p className="text-sm text-github-text-secondary mt-1">Real-time SceneDB telemetry</p>
        </div>
        <div className="ml-auto flex items-center gap-2 text-xs">
          <span className={`w-2 h-2 rounded-full ${connected ? "bg-github-green" : "bg-github-red"}`} />
          <span className="text-github-text-muted">{connected ? "15 fps" : "disconnected"}</span>
        </div>
      </div>

      <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 xl:grid-cols-6 gap-4">
        <StatCard label="Cells" value={stats?.cells ?? alive} subtitle={`${alive} active`} color="blue" />
        <StatCard label="Total Rows" value={stats?.total_rows ?? totalRows} color="green" />
        <StatCard label="Gen Writes" value={stats?.gen_writes ?? 0} subtitle="generation buffer writes" color="yellow" />
        <StatCard label="GPU Buffers" value={stats?.gpu_buffers ?? 0} color="blue" />
        <StatCard label="Pool Free" value={`${poolFree}%`} subtitle="region pool utilization" color={poolFree > 50 ? "green" : poolFree > 25 ? "yellow" : "red"} />
        <StatCard label="Status" value={connected ? "Live" : "—"} subtitle={connected ? `cells: ${alive}` : "offline"} color={connected ? "green" : "red"} />
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        <div className="lg:col-span-2 card">
          <h3 className="card-header">Cells</h3>
          {cells.length === 0 ? (
            <p className="text-sm text-github-text-muted py-8 text-center">No cells registered</p>
          ) : (
            <div className="overflow-x-auto">
              <table className="w-full">
                <thead>
                  <tr>
                    <th className="table-header">ID</th>
                    <th className="table-header">Rows</th>
                    <th className="table-header">Capacity</th>
                    <th className="table-header">Cols</th>
                    <th className="table-header">Utilization</th>
                  </tr>
                </thead>
                <tbody>
                  {cells.slice(0, 8).map((cell) => {
                    const pct = cell.capacity > 0 ? Math.round((cell.rows_in_use / cell.capacity) * 100) : 0;
                    return (
                      <tr key={cell.id} className="hover:bg-github-border-muted/30 transition-colors">
                        <td className="table-cell font-mono text-github-accent">#{cell.id}</td>
                        <td className="table-cell">{cell.rows_in_use.toLocaleString()}</td>
                        <td className="table-cell">{cell.capacity.toLocaleString()}</td>
                        <td className="table-cell">{cell.user_column_count}</td>
                        <td className="table-cell">
                          <div className="flex items-center gap-3">
                            <div className="flex-1 h-1.5 rounded-full bg-github-border-muted overflow-hidden">
                              <div className={`h-full rounded-full transition-all ${pct > 90 ? "bg-github-yellow" : "bg-github-accent"}`} style={{ width: `${pct}%` }} />
                            </div>
                            <span className="text-xs font-mono text-github-text-muted w-10 text-right">{pct}%</span>
                          </div>
                        </td>
                      </tr>
                    );
                  })}
                </tbody>
              </table>
            </div>
          )}
        </div>

        <div className="card space-y-4">
          <h3 className="card-header">Region Pools</h3>
          {pools ? (
            <>
              {pools.row.map((pool, i) => <PoolBar key={`r${i}`} label={`Row class ${i}`} free={pool.free} total={pool.total} />)}
              {pools.slot.map((pool, i) => <PoolBar key={`s${i}`} label={`Slot class ${i}`} free={pool.free} total={pool.total} />)}
            </>
          ) : (
            <p className="text-sm text-github-text-muted py-4 text-center">No pool data</p>
          )}
        </div>
      </div>
    </div>
  );
}

function PoolBar({ label, free, total }: { label: string; free: number; total: number }) {
  const pct = total > 0 ? Math.round((free / total) * 100) : 0;
  return (
    <div className="space-y-1">
      <div className="flex justify-between text-xs">
        <span className="text-github-text-secondary">{label}</span>
        <span className="text-github-text-muted">{free}/{total} free</span>
      </div>
      <div className="h-2 rounded-full bg-github-border-muted overflow-hidden">
        <div className={`h-full rounded-full transition-all ${pct > 50 ? "bg-github-green" : pct > 25 ? "bg-github-yellow" : "bg-github-red"}`} style={{ width: `${pct}%` }} />
      </div>
    </div>
  );
}
