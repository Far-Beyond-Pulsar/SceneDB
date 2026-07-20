"use client";

import { useState } from "react";
import { usePoll, fetchSchema, type TypeSchema } from "@/lib/api";

export default function SchemaPage() {
  const [schema, setSchema] = useState<TypeSchema[]>([]);

  usePoll(async () => setSchema(await fetchSchema()), []);

  return (
    <div className="space-y-6">
      <div>
        <h1 className="text-xl font-semibold">Schema</h1>
        <p className="text-sm text-github-text-secondary mt-1">Registered component type schemas &middot; 15 fps</p>
      </div>

      <div className="card">
        {schema.length === 0 ? (
          <div className="text-center py-12">
            <svg className="w-12 h-12 mx-auto text-github-text-muted mb-4" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={1.5}>
              <path d="M14 2H6a2 2 0 00-2 2v16a2 2 0 002 2h12a2 2 0 002-2V8z" />
              <polyline points="14 2 14 8 20 8" />
            </svg>
            <p className="text-sm text-github-text-muted">No schemas registered</p>
            <p className="text-xs text-github-text-muted mt-2">Schema registration is currently a no-op. It will be populated once TypeToken registration is wired through.</p>
          </div>
        ) : (
          <div className="overflow-x-auto">
            <table className="w-full">
              <thead><tr>
                <th className="table-header">Component ID</th>
                <th className="table-header">Type Name</th>
                <th className="table-header">Size</th>
                <th className="table-header">Alignment</th>
              </tr></thead>
              <tbody>
                {schema.map((s) => (
                  <tr key={s.component_id} className="hover:bg-github-border-muted/30">
                    <td className="table-cell font-mono text-github-accent">C{s.component_id}</td>
                    <td className="table-cell font-mono">{s.type_name}</td>
                    <td className="table-cell font-mono">{s.size} B</td>
                    <td className="table-cell font-mono">{s.align} B</td>
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
