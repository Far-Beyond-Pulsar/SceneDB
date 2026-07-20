"use client";

import { useState, useRef, useEffect, type KeyboardEvent } from "react";

interface LogLine {
  text: string;
  cls: "output" | "input" | "error" | "prompt" | "help" | "info";
}

const HELP_TEXT = [
  "",
  " SceneDB Query Shell",
  "",
  "  cells                        List all registered cells",
  "  cell <id>                    Show cell detail",
  "  cell <id> col <cid>          Show column data for cell",
  "  raw cell=<id> col=<cid>      Query raw row data (optional: start=N end=N)",
  "  stats                        Show aggregate counters",
  "  gpu                          Show GPU store state",
  "  gpu buffers                  Show GPU buffer info",
  "  pools                        Show row/slot region pools",
  "  schema                       Show registered type schemas",
  "  health                       Health check",
  "  help                         This help",
  "  clear                        Clear screen",
  "",
].map((t) => ({ text: t, cls: "help" as const }));

export default function QueryShell() {
  const [logs, setLogs] = useState<LogLine[]>([
    { text: "SceneDB Query Shell — type 'help' for commands", cls: "info" },
  ]);
  const [input, setInput] = useState("");
  const [history, setHistory] = useState<string[]>([]);
  const [histIdx, setHistIdx] = useState(-1);
  const bottomRef = useRef<HTMLDivElement>(null);
  const inputRef = useRef<HTMLInputElement>(null);
  const [busy, setBusy] = useState(false);

  useEffect(() => { bottomRef.current?.scrollIntoView({ behavior: "smooth" }); }, [logs]);
  useEffect(() => { inputRef.current?.focus(); }, []);

  const add = (text: string, cls: LogLine["cls"] = "output") =>
    setLogs((p) => [...p, { text, cls }]);

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

      if (verb === "help") {
        HELP_TEXT.forEach((l) => add(l.text, l.cls));
      } else if (verb === "clear") {
        setLogs([]);
      } else if (verb === "cells") {
        add(await api("/cells"));
      } else if (verb === "stats") {
        add(await api("/stats"));
      } else if (verb === "health") {
        add(await api("/health"));
      } else if (verb === "schema") {
        add(await api("/schema"));
      } else if (verb === "pools") {
        add(await api("/pools"));
      } else if (verb === "gpu") {
        if (parts[1] === "buffers") add(await api("/gpu/buffers"));
        else add(await api("/gpu"));
      } else if (verb === "cell") {
        const id = parts[1];
        if (!id) { add("usage: cell <id> [col <cid>]", "error"); return; }
        if (parts[2] === "col" && parts[3]) {
          add(await api(`/cells/${id}/columns/${parts[3]}`));
        } else {
          add(await api(`/cells/${id}`));
        }
      } else if (verb === "raw") {
        const qp = new URLSearchParams();
        for (let i = 1; i < parts.length; i++) {
          const kv = parts[i].split("=");
          if (kv.length === 2) qp.set(kv[0], kv[1]);
        }
        const qs = qp.toString();
        add(await api(`/query${qs ? "?" + qs : ""}`));
      } else {
        add(`unknown command: ${verb}  (try 'help')`, "error");
      }
    } catch (e: unknown) {
      add(String(e instanceof Error ? e.message : e), "error");
    }
    setBusy(false);
    requestAnimationFrame(() => inputRef.current?.focus());
  };

  const onKeyDown = (e: KeyboardEvent<HTMLInputElement>) => {
    if (e.key === "Enter" && !busy) {
      run(input);
      setInput("");
    } else if (e.key === "ArrowUp") {
      e.preventDefault();
      if (history.length === 0) return;
      const idx = histIdx < 0 ? history.length - 1 : Math.max(0, histIdx - 1);
      setHistIdx(idx);
      setInput(history[idx]);
    } else if (e.key === "ArrowDown") {
      e.preventDefault();
      if (histIdx < 0) return;
      if (histIdx >= history.length - 1) {
        setHistIdx(-1); setInput("");
      } else {
        const idx = histIdx + 1;
        setHistIdx(idx);
        setInput(history[idx]);
      }
    }
  };

  return (
    <div className="card p-0 flex flex-col font-mono text-sm" style={{ height: "70vh" }}>
      <div className="flex-1 overflow-y-auto p-4 space-y-0.5" onClick={() => inputRef.current?.focus()}>
        {logs.map((l, i) => (
          <div key={i} className={`whitespace-pre-wrap break-all ${
            l.cls === "input" ? "text-github-accent" :
            l.cls === "error" ? "text-github-red" :
            l.cls === "help" ? "text-github-text-secondary" :
            l.cls === "info" ? "text-github-text-muted" :
            l.cls === "prompt" ? "text-github-green" :
            "text-github-text"
          }`}>{l.text}</div>
        ))}
        <div ref={bottomRef} />
      </div>
      <div className="flex items-center gap-2 border-t border-github-border-muted px-4 py-2.5 bg-github-card">
        <span className="text-github-green shrink-0">❯</span>
        <input
          ref={inputRef}
          value={input}
          onChange={(e) => setInput(e.target.value)}
          onKeyDown={onKeyDown}
          disabled={busy}
          className="flex-1 bg-transparent outline-none text-github-text placeholder-github-text-muted"
          placeholder={busy ? "running..." : "type a command..."}
        />
      </div>
    </div>
  );
}
