"use client";

import { useState } from "react";
import { usePoll, fetchCells, type CellSnapshot } from "@/lib/api";

export default function CellsPage() {
  const [cells, setCells] = useState<CellSnapshot[]>([]);
  const [sortKey, setSortKey] = useState<keyof CellSnapshot>("id");
  const [sortDir, setSortDir] = useState<"asc" | "desc">("asc");

  usePoll(async () => {
    setCells(await fetchCells());
  }, []);

  const sorted = [...cells].sort((a, b) => {
    const va = a[sortKey], vb = b[sortKey];
    if (typeof va === "number" && typeof vb === "number") {
      return sortDir === "asc" ? va - vb : vb - va;
    }
    return 0;
  });

  const toggleSort = (key: keyof CellSnapshot) => {
    if (key === sortKey) setSortDir((d) => (d === "asc" ? "desc" : "asc"));
    else { setSortKey(key); setSortDir("asc"); }
  };

  const SortIcon = ({ col }: { col: keyof CellSnapshot }) =>
    col === sortKey
      ? <span className="text-github-accent ml-1">{sortDir === "asc" ? "↑" : "↓"}</span>
      : <span className="text-github-text-muted ml-1">↕</span>;

  return (
    <div className="space-y-6">
      <div>
        <h1 className="text-xl font-semibold">Cells</h1>
        <p className="text-sm text-github-text-secondary mt-1">{cells.length} cell{cells.length !== 1 ? "s" : ""} — 15 fps</p>
      </div>

      <div className="card p-0 overflow-hidden">
        <div className="overflow-x-auto">
          <table className="w-full">
            <thead>
              <tr className="border-b border-github-border-muted">
                {(["id", "rows_in_use", "capacity", "user_column_count", "cell_type_name"] as const).map((col) => (
                  <th key={col} className="table-header cursor-pointer select-none hover:text-github-text" onClick={() => toggleSort(col)}>
                    {col.replace(/_/g, " ")} <SortIcon col={col} />
                  </th>
                ))}
                <th className="table-header">Utilization</th>
                <th className="table-header">Columns</th>
              </tr>
            </thead>
            <tbody>
              {sorted.map((cell) => {
                const pct = cell.capacity > 0 ? Math.round((cell.rows_in_use / cell.capacity) * 100) : 0;
                return (
                  <tr key={cell.id} className="hover:bg-github-border-muted/30 transition-colors">
                    <td className="table-cell">
                      <a href={`/cell?id=${cell.id}`} className="font-mono text-github-accent hover:underline">#{cell.id}</a>
                    </td>
                    <td className="table-cell font-mono">{cell.rows_in_use.toLocaleString()}</td>
                    <td className="table-cell font-mono">{cell.capacity.toLocaleString()}</td>
                    <td className="table-cell">{cell.user_column_count}</td>
                    <td className="table-cell">
                      <span className={cell.cell_type_name ? "badge-blue" : "badge-gray"}>
                        {cell.cell_type_name || "unnamed"}
                      </span>
                    </td>
                    <td className="table-cell w-48">
                      <div className="flex items-center gap-3">
                        <div className="flex-1 h-1.5 rounded-full bg-github-border-muted overflow-hidden">
                          <div className={`h-full rounded-full transition-all ${pct > 90 ? "bg-github-yellow" : "bg-github-accent"}`} style={{ width: `${pct}%` }} />
                        </div>
                        <span className="text-xs font-mono text-github-text-muted w-10 text-right">{pct}%</span>
                      </div>
                    </td>
                    <td className="table-cell">
                      <div className="flex flex-wrap gap-1">
                        {cell.pod_columns.map(([cid]) => (
                          <span key={cid} className="badge-blue text-[10px]">C{cid}</span>
                        ))}
                      </div>
                    </td>
                  </tr>
                );
              })}
            </tbody>
          </table>
        </div>
        {cells.length === 0 && <p className="text-sm text-github-text-muted text-center py-12">No cells registered</p>}
      </div>
    </div>
  );
}
