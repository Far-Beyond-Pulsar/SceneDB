"use client";

import { useState, useRef, useEffect, useCallback, type KeyboardEvent, type MouseEvent } from "react";

interface LogLine {
  text: string;
  cls: "output" | "input" | "error" | "prompt" | "help" | "info";
}

const LINE_HEIGHT = 20;
const MINIMAP_BAR = 3;
const MINIMAP_WIDTH = 40;

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

const clsToColor: Record<string, string> = {
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
  const [scrollTop, setScrollTop] = useState(0);
  const scrollRef = useRef<HTMLDivElement>(null);
  const inputRef = useRef<HTMLInputElement>(null);
  const minimapRef = useRef<HTMLDivElement>(null);
  const draggingRef = useRef(false);
  const [busy, setBusy] = useState(false);

  const add = useCallback((text: string, cls: LogLine["cls"] = "output") =>
    setLogs((p) => [...p, { text, cls }]), []);

  const api = async (path: string): Promise<string> => {
    const res = await fetch(`/api${path}`);
    if (!res.ok) throw new Error(`HTTP ${res.status}`);
    return JSON.stringify(await res.json(), null, 2);
  };

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

  const scrollToBottom = () => {
    requestAnimationFrame(() => {
      if (scrollRef.current) scrollRef.current.scrollTop = scrollRef.current.scrollHeight;
    });
  };

  const onKeyDown = (e: KeyboardEvent<HTMLInputElement>) => {
    if (e.key === "Enter" && !busy) { run(input); setInput(""); }
    else if (e.key === "ArrowUp") { e.preventDefault(); if (history.length) { const idx = histIdx < 0 ? history.length - 1 : Math.max(0, histIdx - 1); setHistIdx(idx); setInput(history[idx]); } }
    else if (e.key === "ArrowDown") { e.preventDefault(); if (histIdx >= 0) { if (histIdx >= history.length - 1) { setHistIdx(-1); setInput(""); } else { const idx = histIdx + 1; setHistIdx(idx); setInput(history[idx]); } } }
  };

  const onScroll = useCallback(() => {
    if (scrollRef.current) setScrollTop(scrollRef.current.scrollTop);
  }, []);

  const [viewHeight, setViewHeight] = useState(400);
  useEffect(() => {
    const el = scrollRef.current?.parentElement;
    if (el) setViewHeight(el.clientHeight - 48);
  }, []);

  const totalLines = logs.length;
  const scrollMax = Math.max(totalLines * LINE_HEIGHT - viewHeight, 1);

  const minimapSeek = (clientY: number) => {
    const rect = minimapRef.current?.getBoundingClientRect();
    if (!rect || !scrollRef.current) return;
    const frac = (clientY - rect.top) / rect.height;
    scrollRef.current.scrollTop = Math.round(frac * scrollMax);
  };

  const minimapDown = (e: MouseEvent<HTMLDivElement>) => {
    draggingRef.current = true;
    minimapSeek(e.clientY);
  };

  return (
    <div className="card p-0 flex flex-col font-mono text-sm" style={{ height: "70vh" }}>
      <div className="flex flex-1 min-h-0">
        <div
          ref={scrollRef}
          onScroll={onScroll}
          className="flex-1 overflow-y-auto overscroll-contain"
        >
          <div className="p-4 pb-0 space-y-0.5 select-text" style={{ minHeight: "100%" }} onClick={() => inputRef.current?.focus()}>
            {logs.map((l, i) => (
              <div key={i} style={{ color: clsToColor[l.cls] || "#e6edf3" }} className="whitespace-pre-wrap break-all leading-5">
                {l.text}
              </div>
            ))}
            <div className="h-4" />
          </div>
        </div>

        <div
          ref={minimapRef}
          onMouseDown={minimapDown}
          onMouseMove={(e) => { if (draggingRef.current) minimapSeek(e.clientY); }}
          onMouseUp={() => { draggingRef.current = false; }}
          onMouseLeave={() => { draggingRef.current = false; }}
          className="relative shrink-0 border-l border-github-border-muted cursor-pointer select-none overflow-hidden hover:bg-github-border-muted/20 transition-colors"
          style={{ width: MINIMAP_WIDTH }}
        >
          <div className="absolute inset-0 flex flex-col px-[3px] py-[2px] gap-[1px]">
            {logs.map((l, i) => (
              <div key={i} style={{
                height: 2,
                backgroundColor: clsToColor[l.cls] || "#e6edf3",
                opacity: 0.4,
                width: "100%",
                borderRadius: 1,
                flexShrink: 0,
              }} />
            ))}
          </div>
          <div style={{
            position: "absolute", left: 0, right: 0,
            top: `${(scrollTop / scrollMax) * 100}%`,
            height: `${(viewHeight / (totalLines * LINE_HEIGHT)) * 100}%`,
            backgroundColor: "rgba(88,166,255,0.1)",
            borderLeft: "2px solid rgba(88,166,255,0.4)",
            pointerEvents: "none",
          }} />
        </div>
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
