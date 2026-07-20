"use client";

import { useState } from "react";
import { usePoll, fetchGpu, fetchGpuBuffers, type GpuSnapshot, type GpuBufferSnapshot } from "@/lib/api";

const EMPTY_GPU: GpuSnapshot = { gen_writes: 0, buffers: [], cell_gpu_states: [] };

export default function GpuPage() {
  const [gpu, setGpu] = useState<GpuSnapshot>(EMPTY_GPU);
  const [buffers, setBuffers] = useState<GpuBufferSnapshot[]>([]);

  usePoll(async () => {
    const [g, b] = await Promise.all([fetchGpu(), fetchGpuBuffers()]);
    setGpu(g); setBuffers(b);
  }, []);

  const hasGpu = buffers.length > 0 || gpu.cell_gpu_states.some(s => s.alive || s.dirty_column_count > 0);

  if (!hasGpu) {
    return (
      <div className="space-y-6">
        <div>
          <h1 className="text-xl font-semibold">GPU Store</h1>
        </div>
        <div className="card max-w-xl">
          <div className="flex items-start gap-4 py-4">
            <div className="w-10 h-10 rounded-lg bg-github-blue/10 flex items-center justify-center shrink-0 mt-1">
              <svg className="w-5 h-5 text-github-blue" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2}>
                <rect x="2" y="3" width="20" height="14" rx="2" />
                <rect x="8" y="7" width="8" height="6" rx="1" />
                <circle cx="12" cy="10" r="1.5" fill="currentColor" />
              </svg>
            </div>
            <div>
              <p className="font-medium mb-1">No GPU Store Configured</p>
              <p className="text-sm text-github-text-secondary leading-relaxed">
                The connected SceneDB instance is running in CPU-only mode. GPU telemetry requires
                a <code className="text-github-accent text-xs bg-github-border-muted/50 px-1 rounded">SceneGpuStore</code> —
                enable it by registering cells through the GPU store API and activating the
                <code className="text-github-accent text-xs bg-github-border-muted/50 px-1 rounded ml-1">gpu</code> feature.
              </p>
              <p className="text-sm text-github-text-secondary mt-3">
                The stress TUI (<code className="text-xs bg-github-border-muted/50 px-1 rounded">examples/stress_tui.rs</code>)
                runs CPU workloads only. Switch to a GPU-enabled SceneDB instance to see GPU state here.
              </p>
            </div>
          </div>
        </div>
      </div>
    );
  }

  const activeCells = gpu.cell_gpu_states.filter((s) => s.alive);
  const totalDirty = gpu.cell_gpu_states.reduce((s, c) => s + c.dirty_column_count, 0);
  const totalPending = gpu.cell_gpu_states.reduce((s, c) => s + c.pending_retire_count, 0);

  return (
    <div className="space-y-6">
      <div>
        <h1 className="text-xl font-semibold">GPU Store</h1>
        <p className="text-sm text-github-text-secondary mt-1">
          {buffers.length} buffers &middot; {activeCells.length} GPU-resident cells &middot; {gpu.gen_writes.toLocaleString()} total gen writes &middot; 15 fps
        </p>
      </div>

      <div className="grid grid-cols-1 sm:grid-cols-4 gap-4">
        {[
          { label: "Gen Writes", value: gpu.gen_writes.toLocaleString(), color: "text-github-yellow" as const },
          { label: "GPU Buffers", value: String(buffers.length), color: "text-github-accent" as const },
          { label: "Dirty Columns", value: String(totalDirty), color: totalDirty > 0 ? "text-github-yellow" as const : "text-github-green" as const },
          { label: "Pending Retires", value: String(totalPending), color: totalPending > 0 ? "text-github-red" as const : "text-github-green" as const },
        ].map((s) => (
          <div key={s.label} className="card">
            <p className="stat-label">{s.label}</p>
            <p className={`stat-value ${s.color}`}>{s.value}</p>
          </div>
        ))}
      </div>

      <div className="card">
        <h3 className="card-header">GPU Buffers</h3>
        {buffers.length === 0 ? (
          <p className="text-sm text-github-text-muted text-center py-8">No buffers registered</p>
        ) : (
          <div className="overflow-x-auto">
            <table className="w-full">
              <thead><tr>
                <th className="table-header">Component ID</th>
                <th className="table-header">Element Size</th>
                <th className="table-header">Capacity</th>
                <th className="table-header">Total Bytes</th>
              </tr></thead>
              <tbody>
                {buffers.map((buf) => (
                  <tr key={buf.component_id} className="hover:bg-github-border-muted/30">
                    <td className="table-cell font-mono text-github-accent">C{buf.component_id}</td>
                    <td className="table-cell">{buf.element_size} B</td>
                    <td className="table-cell font-mono">{buf.capacity.toLocaleString()}</td>
                    <td className="table-cell font-mono">{(buf.element_size * buf.capacity).toLocaleString()} B</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        )}
      </div>

      <div className="card">
        <h3 className="card-header">Per-Cell GPU States</h3>
        {gpu.cell_gpu_states.length === 0 ? (
          <p className="text-sm text-github-text-muted text-center py-8">No cells registered</p>
        ) : (
          <div className="overflow-x-auto">
            <table className="w-full">
              <thead><tr>
                <th className="table-header">Cell</th><th className="table-header">Class</th>
                <th className="table-header">Row Base</th><th className="table-header">Slot Base</th>
                <th className="table-header">Slot Cap</th><th className="table-header">Dirty Cols</th>
                <th className="table-header">Pending</th><th className="table-header">Status</th>
              </tr></thead>
              <tbody>
                {gpu.cell_gpu_states.map((s) => (
                  <tr key={s.id} className="hover:bg-github-border-muted/30">
                    <td className="table-cell font-mono text-github-accent">#{s.id}</td>
                    <td className="table-cell">{s.class}</td>
                    <td className="table-cell font-mono">{s.row_base}</td>
                    <td className="table-cell font-mono">{s.slot_base}</td>
                    <td className="table-cell font-mono">{s.slot_capacity}</td>
                    <td className="table-cell"><span className={s.dirty_column_count > 0 ? "badge-yellow" : "badge-green"}>{s.dirty_column_count}</span></td>
                    <td className="table-cell"><span className={s.pending_retire_count > 0 ? "badge-red" : "badge-gray"}>{s.pending_retire_count}</span></td>
                    <td className="table-cell"><span className={s.alive ? "badge-green" : "badge-gray"}>{s.alive ? "Resident" : "Evicted"}</span></td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        )}
      </div>
    </div>
  );
}
