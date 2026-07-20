import { useEffect, useRef } from "react";

const API_BASE = "/api";

export interface TelemetrySnapshot {
  cells: CellSnapshot[];
  gpu: GpuSnapshot;
  schema: TypeSchema[];
  pools: PoolSnapshot;
}

export interface CellSnapshot {
  id: number;
  rows_in_use: number;
  capacity: number;
  user_column_count: number;
  pod_columns: [number, number][];
  generic_columns: [string, number][];
  pod_data: ColumnData[];
  slot_column: number[];
  liveness_bits: number[];
  cell_type_name: string;
}

export interface ColumnData {
  component_id: number;
  element_size: number;
  rows_hex: string[];
}

export interface GpuSnapshot {
  gen_writes: number;
  buffers: GpuBufferSnapshot[];
  cell_gpu_states: CellGpuSnapshot[];
}

export interface GpuBufferSnapshot {
  component_id: number;
  element_size: number;
  capacity: number;
}

export interface CellGpuSnapshot {
  id: number;
  class: number;
  row_base: number;
  slot_base: number;
  slot_capacity: number;
  dirty_column_count: number;
  pending_retire_count: number;
  alive: boolean;
}

export interface TypeSchema {
  component_id: number;
  type_name: string;
  size: number;
  align: number;
}

export interface PoolSnapshot {
  row: PoolInfo[];
  slot: PoolInfo[];
}

export interface PoolInfo {
  region_size: number;
  total: number;
  free: number;
}

export interface StatsSnapshot {
  cells: number;
  cells_alive: number;
  total_rows: number;
  gpu_buffers: number;
  gen_writes: number;
}

async function fetchJson<T>(url: string): Promise<T> {
  const res = await fetch(url);
  if (!res.ok) throw new Error(`HTTP ${res.status}`);
  return res.json();
}

export async function fetchSnapshot(): Promise<TelemetrySnapshot> {
  return fetchJson<TelemetrySnapshot>(`${API_BASE}/`);
}
export async function fetchStats(): Promise<StatsSnapshot> {
  return fetchJson<StatsSnapshot>(`${API_BASE}/stats`);
}
export async function fetchCells(): Promise<CellSnapshot[]> {
  return fetchJson<CellSnapshot[]>(`${API_BASE}/cells`);
}
export async function fetchCell(id: number): Promise<CellSnapshot> {
  return fetchJson<CellSnapshot>(`${API_BASE}/cells/${id}`);
}
export async function fetchGpu(): Promise<GpuSnapshot> {
  return fetchJson<GpuSnapshot>(`${API_BASE}/gpu`);
}
export async function fetchGpuBuffers(): Promise<GpuBufferSnapshot[]> {
  return fetchJson<GpuBufferSnapshot[]>(`${API_BASE}/gpu/buffers`);
}
export async function fetchPools(): Promise<PoolSnapshot> {
  return fetchJson<PoolSnapshot>(`${API_BASE}/pools`);
}
export async function fetchSchema(): Promise<TypeSchema[]> {
  return fetchJson<TypeSchema[]>(`${API_BASE}/schema`);
}

export function usePoll(fn: () => Promise<void>, deps: unknown[] = []) {
  const fnRef = useRef(fn);
  fnRef.current = fn;
  const mountedRef = useRef(true);

  useEffect(() => {
    mountedRef.current = true;
    let timer: ReturnType<typeof setTimeout>;
    const tick = async () => {
      if (!mountedRef.current) return;
      try { await fnRef.current(); } catch { /* keep polling */ }
      if (mountedRef.current) timer = setTimeout(tick, 66);
    };
    tick();
    return () => { mountedRef.current = false; clearTimeout(timer); };
  }, deps);
}
