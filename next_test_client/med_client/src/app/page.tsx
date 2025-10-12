'use client';

import { useMemo, useState } from 'react';
import Viewer from './Viewer/viewer';

type ApiResponse = {
  message: string;
  detections: number;
  chosen_detection?: { box: [number, number, number, number]; confidence: number; class_id: number };
  result?: { lot_number: string | null; expiry_date: string | null };
  artifacts?: { full_frame_path?: string; crop_path?: string; final_path?: string } | null;
};

export default function Home() {
  const [endpoint, setEndpoint] = useState('http://localhost:8000/process?save_artifacts=true');
  const [data, setData] = useState<ApiResponse | null>(null);
  const [raw, setRaw] = useState('');
  const [error, setError] = useState('');
  const [loading, setLoading] = useState(false);
  const [lightboxUrl, setLightboxUrl] = useState<string | null>(null);

  const apiBase = useMemo(() => {
    try { const u = new URL(endpoint); return `${u.protocol}//${u.host}`; }
    catch { return 'http://localhost:8000'; }
  }, [endpoint]);

  const toRunsUrl = (p?: string | null) => {
    if (!p) return undefined;
    const i = p.lastIndexOf('/runs/');
    if (i === -1) return undefined;
    const sub = p.substring(i + '/runs/'.length).replaceAll('\\', '/');
    return `${apiBase}/runs/${sub}?t=${Date.now()}`;
  };

  const handleSubmit = async () => {
    setLoading(true); setError(''); setData(null); setRaw('');
    try {
      const res = await fetch(endpoint, { method: 'POST', headers: { 'Content-Type': 'application/json' } });
      const json: ApiResponse = await res.json();
      setData(json);
      setRaw(JSON.stringify(json, null, 2));
      if (!res.ok) setError(`HTTP ${res.status}: ${json?.message ?? 'Request failed'}`);
    } catch (e: any) { setError(e?.message ?? 'Request failed'); }
    finally { setLoading(false); }
  };

  const fullUrl  = toRunsUrl(data?.artifacts?.full_frame_path);
  const cropUrl  = toRunsUrl(data?.artifacts?.crop_path);
  const finalUrl = toRunsUrl(data?.artifacts?.final_path);

  return (
    <div className="font-sans min-h-screen bg-gray-50 text-gray-900">
      {/* Top bar */}
      <div className="h-10 px-2 py-1 flex items-center gap-2 border-b sticky top-0 bg-gray-50 z-10">
        <h1 className="text-[12px] font-semibold mr-2">Medicine OCR</h1>
        <input
          className="flex-1 rounded-md border px-2 py-1 text-[11px] outline-none focus:ring-1 focus:ring-blue-500"
          value={endpoint}
          onChange={(e) => setEndpoint(e.target.value)}
          placeholder="http://localhost:8000/process?save_artifacts=true"
        />
        <button
          onClick={handleSubmit}
          disabled={loading}
          className="rounded-md px-3 py-1 text-[11px] bg-blue-600 text-white hover:bg-blue-700 disabled:opacity-50"
        >
          {loading ? 'Processing…' : 'Submit'}
        </button>
      </div>

      <main className="px-2 py-2">
        <div className="grid grid-cols-12 gap-3">
          {/* LEFT: 50% player / 50% thumbnails row */}
          <div className="col-span-8 lg:col-span-9">
            {/* Reference height so percentages apply; page can scroll */}
            <div className="grid gap-2 h-[110vh] grid-rows-[50%_50%]">
              {/* Player (clipped, overlay visible) */}
              <Viewer />

              {/* Thumbnails row: three equal cards, each clips its image */}
              <div className="grid grid-cols-3 gap-2 h-full">
                {[
                  { url: fullUrl,  label: 'Full'  },
                  { url: cropUrl,  label: 'Crop'  },
                  { url: finalUrl, label: 'Final' },
                ].map(({ url, label }) => (
                  <button
                    key={label}
                    type="button"
                    onClick={() => url && setLightboxUrl(url)}
                    className="group h-full w-full"
                    title={label}
                  >
                    <div className="h-full w-full rounded-md border overflow-hidden bg-black">
                      {url ? (
                        <img
                          src={url}
                          alt={label}
                          className="block w-full h-full object-contain transition-transform group-hover:scale-[1.01]"
                        />
                      ) : (
                        <span className="flex items-center justify-center w-full h-full text-[11px] text-gray-400">
                          {label}: n/a
                        </span>
                      )}
                    </div>
                  </button>
                ))}
              </div>
            </div>
          </div>

          {/* RIGHT: results (scrolls independently if tall) */}
          <div className="col-span-4 lg:col-span-3 rounded-md border bg-white p-2 text-[11px] leading-tight max-h-[110vh] overflow-auto">
          {error && (
            <div className="mb-2 rounded-md bg-red-50 border border-red-200 px-2 py-1 text-red-700">{error}</div>
          )}

          {data ? (
            <div className="space-y-2">
              {/* Header */}
              <div className="flex items-center justify-between">
                <div className="text-[12px] font-medium">Result</div>
                <span className="rounded-full bg-gray-100 px-2 py-[2px] text-[10px]">{data.message}</span>
              </div>

              {/* === Predictions / Confidence / Stats (added back) === */}
              <section className="rounded-md border p-2 space-y-2">
                <div className="text-[11px] font-medium">Predictions</div>

                <div className="grid grid-cols-2 gap-2">
                  <div className="rounded border p-2">
                    <div className="text-[10px] text-gray-500">Detections</div>
                    <div className="text-sm font-semibold">{data.detections ?? 0}</div>
                  </div>

                  <div className="rounded border p-2">
                    <div className="text-[10px] text-gray-500">Confidence</div>
                    <div className="text-sm font-semibold">
                      {data?.chosen_detection?.confidence !== undefined
                        ? `${Math.round(data.chosen_detection.confidence * 100)}%`
                        : '—'}
                    </div>
                  </div>

                  <div className="rounded border p-2">
                    <div className="text-[10px] text-gray-500">Class ID</div>
                    <div className="text-sm font-semibold">
                      {data?.chosen_detection?.class_id ?? '—'}
                    </div>
                  </div>

                  <div className="rounded border p-2 col-span-2">
                    <div className="text-[10px] text-gray-500">Box</div>
                    <div className="font-mono text-[10px]">
                      {data?.chosen_detection?.box
                        ? `[${data.chosen_detection.box.join(', ')}]`
                        : '—'}
                    </div>
                  </div>
                </div>

                {/* Stats row (extend if backend returns more like processing_ms, model, etc.) */}
                <div className="grid grid-cols-2 gap-2">
                  <div className="rounded border p-2">
                    <div className="text-[10px] text-gray-500">Lot</div>
                    <div className="text-sm font-semibold">{data?.result?.lot_number ?? '—'}</div>
                  </div>
                  <div className="rounded border p-2">
                    <div className="text-[10px] text-gray-500">Expiry</div>
                    <div className="text-sm font-semibold">{data?.result?.expiry_date ?? '—'}</div>
                  </div>
                </div>
              </section>

              {/* Artifacts list (optional; server paths) */}
              {data.artifacts && (
                <section className="rounded-md border p-2 space-y-1">
                  <div className="text-[11px] font-medium">Artifacts</div>
                  <ul className="list-disc ml-4 font-mono text-[10px] break-all space-y-[2px]">
                    {data.artifacts.full_frame_path && <li>full: {data.artifacts.full_frame_path}</li>}
                    {data.artifacts.crop_path && <li>crop: {data.artifacts.crop_path}</li>}
                    {data.artifacts.final_path && <li>final: {data.artifacts.final_path}</li>}
                  </ul>
                </section>
              )}

              {/* Raw JSON (stays last) */}
              <details className="rounded-md border p-2">
                <summary className="cursor-pointer select-none text-[10px]">Raw JSON</summary>
                <pre className="mt-2 text-[10px] bg-gray-50 p-2 rounded overflow-auto max-h-60">
                  {JSON.stringify(data, null, 2)}
                </pre>
              </details>
            </div>
          ) : (
            <div className="text-gray-500 text-[12px]">Run a request to see results.</div>
          )}
          </div>
        </div>
      </main>

      {/* Fullscreen lightbox */}
      {lightboxUrl && (
        <button
          type="button"
          onClick={() => setLightboxUrl(null)}
          className="fixed inset-0 z-50 bg-black/80 flex items-center justify-center p-4"
          title="Close"
        >
          <img src={lightboxUrl} alt="Preview" className="max-w-[95vw] max-h-[95vh] object-contain rounded-md shadow-2xl" />
        </button>
      )}
    </div>
  );
}
