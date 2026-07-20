"use client";

import { useState, useRef, useEffect, useCallback, type KeyboardEvent, type MouseEvent } from "react";

interface LogLine {
  text: string;
  cls: "output" | "input" | "error" | "prompt" | "help" | "info";
}

const MINIMAP_W = 120;
const FONT_PX = 7;

const HELP_TEXT: LogLine[] = [
  { text: "", cls: "help" }, { text: " SceneDB Query Shell", cls: "help" }, { text: "", cls: "help" },
  { text: "  cells                        List all cells", cls: "help" },
  { text: "  cell <id>                    Cell detail", cls: "help" },
  { text: "  cell <id> col <cid>          Column data", cls: "help" },
  { text: "  raw cell=<id> col=<cid>      Raw row data", cls: "help" },
  { text: "  stats / gpu / pools / schema  System state", cls: "help" },
  { text: "  health                       Health check", cls: "help" },
  { text: "  help / clear                 This / clear", cls: "help" },
  { text: "", cls: "help" },
];

const CLS_COLOR: Record<string, string> = {
  input: "#58a6ff", error: "#f85149", help: "#9198a1",
  info: "#6e7681", prompt: "#3fb950", output: "#f0f6fc",
};

function drawMinimap(canvas: HTMLCanvasElement, logs: LogLine[], vpFrac: number, vpSize: number) {
  const dpr = window.devicePixelRatio || 1;
  const w = MINIMAP_W;
  const h = canvas.parentElement!.clientHeight;
  canvas.width = w * dpr;
  canvas.height = h * dpr;
  canvas.style.width = w + "px";
  canvas.style.height = h + "px";

  const ctx = canvas.getContext("2d")!;
  ctx.scale(dpr, dpr);
  ctx.fillStyle = "#0d1117";
  ctx.fillRect(0, 0, w, h);

  if (logs.length === 0) return;

  const lineH = FONT_PX;
  const totalH = logs.length * lineH;
  const s = Math.min(1, h / Math.max(totalH, h));

  ctx.save();
  ctx.translate(0, 0);
  ctx.scale(1, s);
  ctx.font = `${FONT_PX}px ui-monospace,SFMono-Regular,Menlo,Monaco,Consolas,monospace`;
  ctx.textBaseline = "top";
  for (let i = 0; i < logs.length; i++) {
    const l = logs[i];
    ctx.fillStyle = CLS_COLOR[l.cls] || "#f0f6fc";
    ctx.globalAlpha = 0.5;
    ctx.fillText(l.text.slice(0, 28), 2, i * lineH);
  }
  ctx.globalAlpha = 1;
  ctx.restore();

  // viewport
  const viewportTop = vpFrac * h;
  const viewportH = Math.max(vpSize * h, 4);
  ctx.fillStyle = "rgba(88,166,255,0.06)";
  ctx.fillRect(0, viewportTop, w, viewportH);
  ctx.strokeStyle = "rgba(88,166,255,0.3)";
  ctx.lineWidth = 1;
  ctx.beginPath();
  ctx.moveTo(0, viewportTop); ctx.lineTo(w, viewportTop); ctx.stroke();
  ctx.beginPath();
  ctx.moveTo(0, viewportTop + viewportH); ctx.lineTo(w, viewportTop + viewportH); ctx.stroke();
}

export default function QueryShell() {
  const [logs, setLogs] = useState<LogLine[]>([
    { text: "SceneDB Query Shell — type 'help' for commands", cls: "info" },
  ]);
  const [input, setInput] = useState("");
  const [history, setHistory] = useState<string[]>([]);
  const [histIdx, setHistIdx] = useState(-1);
  const [busy, setBusy] = useState(false);
  const scrollRef = useRef<HTMLDivElement>(null);
  const inputRef = useRef<HTMLInputElement>(null);
  const minimapRef = useRef<HTMLDivElement>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const dragging = useRef(false);
  const drawTicket = useRef(0);

  const scheduleDraw = useCallback(() => {
    drawTicket.current += 1;
    const ticket = drawTicket.current;
    requestAnimationFrame(() => {
      if (ticket !== drawTicket.current) return;
      const c = canvasRef.current, s = scrollRef.current;
      if (!c || !s) return;
      const max = s.scrollHeight - s.clientHeight;
      const vpFrac = max > 0 ? s.scrollTop / max : 0;
      const vpSize = s.clientHeight / s.scrollHeight;
      drawMinimap(c, logs, vpFrac, vpSize);
    });
  }, [logs]);

  useEffect(() => {
    const el = scrollRef.current;
    if (!el) return;
    scheduleDraw();
    const onScroll = () => scheduleDraw();
    el.addEventListener("scroll", onScroll, { passive: true });
    return () => el.removeEventListener("scroll", onScroll);
  }, [scheduleDraw]);

  const add = useCallback((text: string, cls: LogLine["cls"] = "output") => {
    if (text.includes("\n")) {
      setLogs((p) => [...p, ...text.split("\n").map((t) => ({ text: t, cls }))]);
    } else {
      setLogs((p) => [...p, { text, cls }]);
    }
  }, []);

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
    const max = s.scrollHeight - s.clientHeight;
    if (max <= 0) return;
    s.scrollTop = Math.max(0, Math.min(max, ((clientY - r.top) / r.height) * max));
  };

  const onMinimapDown = (e: MouseEvent) => { dragging.current = true; seek(e.clientY); };
  const onMinimapMove = (e: MouseEvent) => { if (dragging.current) seek(e.clientY); };
  const onMinimapUp = () => { dragging.current = false; };

  return (
    <div className="card p-0 flex flex-col font-mono text-sm overflow-hidden relative" style={{ height: "70vh" }}>
      <div ref={scrollRef} className="flex-1 overflow-y-auto">
        <div className="p-4 pb-0 space-y-0.5 select-text" onClick={() => inputRef.current?.focus()}>
          {logs.map((l, i) => (
            <div key={i} style={{ color: CLS_COLOR[l.cls] || "#f0f6fc" }} className="whitespace-pre-wrap break-all leading-5">
              {l.text}
            </div>
          ))}
          <div className="h-4" />
        </div>
      </div>

      <div
        ref={minimapRef}
        onMouseDown={onMinimapDown}
        onMouseMove={onMinimapMove}
        onMouseUp={onMinimapUp}
        onMouseLeave={onMinimapUp}
        className="absolute right-0 top-0 bottom-12 cursor-pointer select-none z-10 border-l border-github-border-muted"
      >
        <canvas ref={canvasRef} className="block" />
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
