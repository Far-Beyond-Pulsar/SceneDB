"use client";

import { useState } from "react";
import { usePoll, fetchPools, type PoolSnapshot } from "@/lib/api";

export default function PoolsPage() {
  const [pools, setPools] = useState<PoolSnapshot | null>(null);

  usePoll(async () => setPools(await fetchPools()), []);

  if (!pools) return <p className="text-github-text-muted text-center py-20">Loading...</p>;

  return (
    <div className="space-y-6">
      <div>
        <h1 className="text-xl font-semibold">Region Pools</h1>
        <p className="text-sm text-github-text-secondary mt-1">
          {pools.row.length} row pool{pools.row.length !== 1 ? "s" : ""} &middot; {pools.slot.length} slot pool{pools.slot.length !== 1 ? "s" : ""} &middot; 15 fps
        </p>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        <div className="card">
          <h3 className="card-header">Row Region Pools</h3>
          {pools.row.length === 0 ? (
            <p className="text-sm text-github-text-muted text-center py-8">No row pools configured</p>
          ) : (
            <div className="space-y-5">
              {pools.row.map((pool, i) => {
                const used = pool.total - pool.free;
                const pct = pool.total > 0 ? Math.round((pool.free / pool.total) * 100) : 0;
                return (
                  <div key={i}>
                    <div className="flex items-center justify-between mb-2">
                      <div>
                        <p className="text-sm font-medium">Class {i} &middot; <span className="font-mono">{pool.region_size}</span></p>
                      </div>
                      <span className="text-sm font-mono tabular-nums">{pool.free} / {pool.total} free</span>
                    </div>
                    <div className="h-3 rounded-full bg-github-border-muted overflow-hidden">
                      <div className={`h-full rounded-full transition-all ${pct > 50 ? "bg-github-green" : pct > 25 ? "bg-github-yellow" : "bg-github-red"}`} style={{ width: `${pct}%` }} />
                    </div>
                    <div className="flex justify-between text-xs text-github-text-muted mt-1">
                      <span>{used} used</span>
                      <span>{pct}% free</span>
                    </div>
                  </div>
                );
              })}
            </div>
          )}
        </div>

        <div className="card">
          <h3 className="card-header">Slot Region Pools</h3>
          {pools.slot.length === 0 ? (
            <p className="text-sm text-github-text-muted text-center py-8">No slot pools configured</p>
          ) : (
            <div className="space-y-5">
              {pools.slot.map((pool, i) => {
                const used = pool.total - pool.free;
                const pct = pool.total > 0 ? Math.round((pool.free / pool.total) * 100) : 0;
                return (
                  <div key={i}>
                    <div className="flex items-center justify-between mb-2">
                      <div>
                        <p className="text-sm font-medium">Class {i} &middot; <span className="font-mono">{pool.region_size}</span></p>
                      </div>
                      <span className="text-sm font-mono tabular-nums">{pool.free} / {pool.total} free</span>
                    </div>
                    <div className="h-3 rounded-full bg-github-border-muted overflow-hidden">
                      <div className={`h-full rounded-full transition-all ${pct > 50 ? "bg-github-green" : pct > 25 ? "bg-github-yellow" : "bg-github-red"}`} style={{ width: `${pct}%` }} />
                    </div>
                    <div className="flex justify-between text-xs text-github-text-muted mt-1">
                      <span>{used} used</span>
                      <span>{pct}% free</span>
                    </div>
                  </div>
                );
              })}
            </div>
          )}
        </div>
      </div>
    </div>
  );
}
