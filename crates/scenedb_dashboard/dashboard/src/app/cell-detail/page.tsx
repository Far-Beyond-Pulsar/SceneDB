"use client";

import { Suspense, useState } from "react";
import { useSearchParams } from "next/navigation";
import { usePoll, fetchCell, fetchGpu, type CellSnapshot, type CellGpuSnapshot } from "@/lib/api";

export default function CellDetailWrapper() {
  return (
    <Suspense fallback={<p className="text-github-text-muted text-center py-20">Loading...</p>}>
      <CellDetail />
    </Suspense>
  );
}

function CellDetail() {
  const searchParams = useSearchParams();
  const cellId = parseInt(searchParams.get("id") || "-1", 10);
  const [cell, setCell] = useState<CellSnapshot | null>(null);
  const [gpuState, setGpuState] = useState<CellGpuSnapshot | null>(null);
  const [error, setError] = useState<string | null>(null);

  usePoll(async () => {
    if (cellId < 0) return;
    try {
      const [c, g] = await Promise.all([fetchCell(cellId), fetchGpu()]);
      setCell(c);
      setGpuState(g.cell_gpu_states.find((s) => s.id === cellId) ?? null);
      setError(null);
    } catch (e) { setError(String(e)); }
  }, [cellId]);

  if (cellId < 0)
    return <div className="text-center py-20"><p className="text-github-text-secondary">No cell selected. <a href="/cells" className="text-github-accent hover:underline">View all cells</a></p></div>;

  if (error && !cell) return <div className="text-center py-20"><p className="text-github-red">{error}</p></div>;
  if (!cell) return <p className="text-github-text-muted text-center py-20">Loading...</p>;

  const pct = cell.capacity > 0 ? Math.round((cell.rows_in_use / cell.capacity) * 100) : 0;
  const liveBits = cell.liveness_bits.reduce((s, w) => s + popcount64(w), 0);

  return (
    <div className="space-y-6 max-w-5xl">
      <div className="flex items-center gap-4">
        <div>
          <h1 className="text-xl font-semibold">Cell <span className="font-mono text-github-accent">#{cell.id}</span></h1>
          <p className="text-sm text-github-text-muted">{cell.cell_type_name || "unnamed"} &middot; {cell.user_column_count} cols</p>
        </div>
        <div className="ml-auto flex items-center gap-2">
          <span className={pct > 90 ? "badge-yellow" : "badge-green"}>{pct}% full</span>
          {gpuState && <span className={gpuState.alive ? "badge-green" : "badge-gray"}>{gpuState.alive ? "GPU resident" : "evicted"}</span>}
        </div>
      </div>

      <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
        <div className="card"><p className="card-header">Storage</p>
          <DetailRow label="Rows" value={cell.rows_in_use.toLocaleString()} />
          <DetailRow label="Capacity" value={cell.capacity.toLocaleString()} />
          <DetailRow label="Columns" value={String(cell.user_column_count)} />
          <DetailRow label="Type" value={cell.cell_type_name || "—"} />
        </div>
        <div className="card"><p className="card-header">Liveness</p>
          <DetailRow label="Live" value={liveBits.toLocaleString()} />
          <DetailRow label="Dead" value={(cell.liveness_bits.length * 64 - liveBits).toLocaleString()} />
          <DetailRow label="Words" value={String(cell.liveness_bits.length)} />
        </div>
        {gpuState && <div className="card"><p className="card-header">GPU</p>
          <DetailRow label="Class" value={String(gpuState.class)} />
          <DetailRow label="Row base" value={String(gpuState.row_base)} />
          <DetailRow label="Slot base" value={String(gpuState.slot_base)} />
          <DetailRow label="Dirty cols" value={String(gpuState.dirty_column_count)} />
          <DetailRow label="Pending" value={String(gpuState.pending_retire_count)} />
        </div>}
      </div>

      {cell.pod_data.length > 0 && (
        <div className="card"><h3 className="card-header">Columns</h3>
          <div className="overflow-x-auto"><table className="w-full">
            <thead><tr><th className="table-header">Component</th><th className="table-header">Size</th><th className="table-header">Rows</th><th className="table-header">Preview</th></tr></thead>
            <tbody>{cell.pod_data.map((col) => (
              <tr key={col.component_id} className="hover:bg-github-border-muted/30">
                <td className="table-cell font-mono text-github-accent">C{col.component_id}</td>
                <td className="table-cell">{col.element_size} B</td>
                <td className="table-cell">{col.rows_hex.length}</td>
                <td className="table-cell font-mono text-xs text-github-text-secondary max-w-[200px] truncate">{col.rows_hex[0]?.slice(0, 32) || "—"}</td>
              </tr>
            ))}</tbody>
          </table></div>
        </div>
      )}

      {cell.liveness_bits.length > 0 && (
        <div className="card"><h3 className="card-header">Liveness</h3>
          <p className="text-xs text-github-text-muted mb-3">Green = live, dark = dead. Up to 1024 rows.</p>
          <LivenessHeatmap words={cell.liveness_bits} maxRows={1024} />
        </div>
      )}
    </div>
  );
}

function DetailRow({ label, value }: { label: string; value: string }) {
  return <div className="flex justify-between text-sm"><span className="text-github-text-secondary">{label}</span><span className="font-mono text-github-text">{value}</span></div>;
}

function popcount64(w: number): number {
  const lo = w >>> 0, hi = Math.floor(w / 0x100000000) >>> 0;
  return popcount32(lo) + popcount32(hi);
}
function popcount32(n: number): number {
  n = n - ((n >>> 1) & 0x55555555);
  n = (n & 0x33333333) + ((n >>> 2) & 0x33333333);
  n = (n + (n >>> 4)) & 0x0f0f0f0f;
  return (n * 0x01010101) >>> 24;
}

function LivenessHeatmap({ words, maxRows }: { words: number[]; maxRows: number }) {
  const cells: boolean[] = [];
  for (let wi = 0; wi < words.length && cells.length < maxRows; wi++) {
    const w = words[wi], lo = w >>> 0, hi = Math.floor(w / 0x100000000) >>> 0;
    for (let bi = 0; bi < 32 && cells.length < maxRows; bi++) cells.push(!!(lo & (1 << bi)));
    for (let bi = 0; bi < 32 && cells.length < maxRows; bi++) cells.push(!!(hi & (1 << bi)));
  }
  const cols = 64, rows = Math.ceil(cells.length / cols);
  return (
    <div className="inline-flex flex-col gap-[2px]">
      {Array.from({ length: rows }, (_, ri) => (
        <div key={ri} className="flex gap-[2px]">
          {Array.from({ length: cols }, (_, ci) => {
            const idx = ri * cols + ci;
            return <div key={ci} className={`w-[6px] h-[6px] rounded-[1px] ${idx >= cells.length ? "bg-transparent" : cells[idx] ? "bg-github-green" : "bg-github-border-muted"}`} title={`Row ${idx}: ${cells[idx] ? "live" : "dead"}`} />;
          })}
        </div>
      ))}
    </div>
  );
}
