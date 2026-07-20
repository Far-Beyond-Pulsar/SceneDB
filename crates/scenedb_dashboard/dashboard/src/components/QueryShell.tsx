"use client";

import { useState, useRef, useEffect, useCallback, type KeyboardEvent, type MouseEvent } from "react";

interface LogLine {
  text: string;
  cls: "output" | "input" | "error" | "prompt" | "help" | "info";
}

const MINIMAP_W = 36;
const BAR_H = 2;
const BAR_GAP = 1;

const HELP_TEXT: LogLine[] = [
  { text: "", cls: "help" },
  { text: " SceneDB Query Shell", cls: "help" },
  { text: "", cls: "help" },
  { text: "  cells                        List all registered cells", cls: "help" },
  { text: "  cell <id>                    Show cell detail", cls: "help" },
  { text: "  cell <id> col <cid>          Show column data for cell", cls: "help" },
  { text: "  raw cell=<id> col=<cid>      Query raw row data (optional: start=N end=N)", cls: "help" },
  { text: "  stats                        Show aggregate counters", cls: "help" },
  { text: "  gpu                          Show GPU store state", cls: "help" },
  { text: "  gpu buffers                  Show GPU buffer info", cls: "help" },
  { text: "  pools                        Show row/slot region pools", cls: "help" },
  { text: "  schema                       Show registered type schemas", cls: "help" },
  { text: "  health                       Health check", cls: "help" },
  { text: "  help                         This help", cls: "help" },
  { text: "  clear                        Clear screen", cls: "help" },
  { text: "", cls: "help" },
];

const CLS_COLOR: Record<string, string> = {
  input: "#58a6ff",
  error: "#f85149",
  help: "#8b949e",
  info: "#6e7681",
  prompt: "#3fb950",
  output: "#e6edf3",
};

export default function QueryShell() {
  const [logs, setLogs] = useState<LogLine[]>([
    { text: "SceneDB Query Shell — type 'help' for commands", cls: "info" },
  ]);
  const [input, setInput] = useState("");
  const [history, setHistory] = useState<string[]>([]);
  const [histIdx, setHistIdx] = useState(-1);
  const scrollRef = useRef<HTMLDivElement>(null);
  const inputRef = useRef<HTMLInputElement>(null);
  const minimapRef = useRef<HTMLDivElement>(null);
  const dragging = useRef(false);
  const [busy, setBusy] = useState(false);

  const add = useCallback((text: string, cls: LogLine["cls"] = "output") =>
    setLogs((p) => [...p, { text, cls }]), []);

  const api = async (path: string): Promise<string> => {
    const res = await fetch(`/api${path}`);
    if (!res.ok) throw new Error(`HTTP ${res.status}`);
    return JSON.stringify(await res.json(), null, 2);
  };

  const scrollToBottom = () => requestAnimationFrame(() => {
    if (scrollRef.current) scrollRef.current.scrollTop = scrollRef.current.scrollHeight;
  });

  const run = async (cmd: string) => {
    const trimmed = cmd.trim();
    if (!trimmed) return;
    add(`$ ${trimmed}`, "input");
    setHistory((p) => [...p, trimmed]);
    setHistIdx(-1);
    setBusy(true);
    try {
      const parts = trimmed.split(/\s+/);
      const verb = parts[0].toLowerCase();
      if (verb === "help") HELP_TEXT.forEach((l) => add(l.text, l.cls));
      else if (verb === "clear") setLogs([]);
      else if (verb === "cells") add(await api("/cells"));
      else if (verb === "stats") add(await api("/stats"));
      else if (verb === "health") add(await api("/health"));
      else if (verb === "schema") add(await api("/schema"));
      else if (verb === "pools") add(await api("/pools"));
      else if (verb === "gpu") add(parts[1] === "buffers" ? await api("/gpu/buffers") : await api("/gpu"));
      else if (verb === "cell") {
        if (!parts[1]) { add("usage: cell <id> [col <cid>]", "error"); return; }
        add(parts[2] === "col" && parts[3] ? await api(`/cells/${parts[1]}/columns/${parts[3]}`) : await api(`/cells/${parts[1]}`));
      } else if (verb === "raw") {
        const qp = new URLSearchParams();
        for (let i = 1; i < parts.length; i++) { const kv = parts[i].split("="); if (kv.length === 2) qp.set(kv[0], kv[1]); }
        add(await api(`/query${qp.toString() ? "?" + qp.toString() : ""}`));
      } else add(`unknown command: ${verb}  (try 'help')`, "error");
    } catch (e: unknown) {
      add(String(e instanceof Error ? e.message : e), "error");
    }
    setBusy(false);
    requestAnimationFrame(() => { inputRef.current?.focus(); scrollToBottom(); });
  };

  const onKeyDown = (e: KeyboardEvent<HTMLInputElement>) => {
    if (e.key === "Enter" && !busy) { run(input); setInput(""); }
    else if (e.key === "ArrowUp") { e.preventDefault(); if (history.length) { const idx = histIdx < 0 ? history.length - 1 : Math.max(0, histIdx - 1); setHistIdx(idx); setInput(history[idx]); } }
    else if (e.key === "ArrowDown") { e.preventDefault(); if (histIdx >= 0) { if (histIdx >= history.length - 1) { setHistIdx(-1); setInput(""); } else { const idx = histIdx + 1; setHistIdx(idx); setInput(history[idx]); } } }
  };

  const seek = (clientY: number) => {
    const r = minimapRef.current?.getBoundingClientRect();
    const s = scrollRef.current;
    if (!r || !s) return;
    const maxScroll = s.scrollHeight - s.clientHeight;
    if (maxScroll <= 0) return;
    const frac = Math.max(0, Math.min(1, (clientY - r.top) / r.height));
    s.scrollTop = frac * maxScroll;
  };

  const onMinimapDown = (e: MouseEvent<HTMLDivElement>) => { dragging.current = true; seek(e.clientY); };
  const onMinimapMove = (e: MouseEvent<HTMLDivElement>) => { if (dragging.current) seek(e.clientY); };
  const onMinimapUp = () => { dragging.current = false; };

  return (
    <div className="card p-0 flex flex-col font-mono text-sm overflow-hidden relative" style={{ height: "70vh" }}>
      <div ref={scrollRef} className="flex-1 overflow-y-auto">
        <div className="p-4 pb-0 space-y-0.5 select-text" onClick={() => inputRef.current?.focus()}>
          {logs.map((l, i) => (
            <div key={i} style={{ color: CLS_COLOR[l.cls] || "#e6edf3" }} className="whitespace-pre-wrap break-all leading-5">
              {l.text}
            </div>
          ))}
          <div className="h-4" />
        </div>
      </div>

      {/* minimap overlay */}
      <div
        ref={minimapRef}
        onMouseDown={onMinimapDown}
        onMouseMove={onMinimapMove}
        onMouseUp={onMinimapUp}
        onMouseLeave={onMinimapUp}
        className="absolute right-0 top-0 bottom-12 cursor-pointer select-none z-10"
        style={{ width: MINIMAP_W }}
      >
        <MiniMapBars logs={logs} scrollRef={scrollRef} />
      </div>

      <div className="flex items-center gap-2 border-t border-github-border-muted px-4 py-2.5 bg-github-card shrink-0">
        <span className="text-github-green shrink-0">❯</span>
        <input
          ref={inputRef}
          value={input}
          onChange={(e) => setInput(e.target.value)}
          onKeyDown={onKeyDown}
          disabled={busy}
          className="flex-1 bg-transparent outline-none text-github-text placeholder-github-text-muted"
          placeholder={busy ? "running…" : "type a command…"}
        />
      </div>
    </div>
  );
}

function MiniMapBars({ logs, scrollRef }: { logs: LogLine[]; scrollRef: React.RefObject<HTMLDivElement | null> }) {
  const [vp, setVp] = useState({ top: 0, height: 10 });

  useEffect(() => {
    const el = scrollRef.current;
    if (!el) return;
    const update = () => {
      const max = el.scrollHeight - el.clientHeight;
      const top = max > 0 ? (el.scrollTop / max) * 100 : 0;
      const height = Math.max((el.clientHeight / el.scrollHeight) * 100, 2);
      setVp({ top, height });
    };
    update();
    el.addEventListener("scroll", update, { passive: true });
    return () => el.removeEventListener("scroll", update);
  }, [logs.length, scrollRef]);

  const barH = BAR_H + BAR_GAP;
  const totalH = logs.length * barH;

  return (
    <div className="relative w-full h-full">
      <div className="absolute inset-0 flex flex-col pt-2" style={{ gap: BAR_GAP }}>
        {logs.map((l, i) => (
          <div key={i} style={{
            height: BAR_H,
            backgroundColor: CLS_COLOR[l.cls] || "#e6edf3",
            opacity: 0.3,
            flexShrink: 0,
            borderRadius: 1,
            marginLeft: 4,
            marginRight: 4,
          }} />
        ))}
      </div>
      <div style={{
        position: "absolute", left: 1, right: 1,
        top: `${vp.top}%`,
        height: `${vp.height}%`,
        backgroundColor: "rgba(88,166,255,0.08)",
        borderLeft: "2px solid rgba(88,166,255,0.35)",
        borderRadius: 2,
        pointerEvents: "none",
      }} />
    </div>
  );
}


