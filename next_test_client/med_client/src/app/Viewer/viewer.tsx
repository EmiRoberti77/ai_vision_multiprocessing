'use client';
import { useState } from 'react';

export default function Viewer() {
  const [src] = useState('http://localhost:8000/stream/mjpg?fps=15&quality=80');

  return (
    <div className="relative w-full h-full rounded-md border overflow-hidden bg-black">
      {/* Live image */}
      <img
        src={src}
        alt="Live stream"
        className="absolute inset-0 block w-full h-full object-contain select-none"
      />
      {/* Overlay label - always above the image */}
      <div className="absolute top-2 left-2 z-20 px-2 py-1 rounded bg-black/65 text-white text-[11px]">
        OAIX_MEDIA_SERVER v1.0
      </div>
    </div>
  );
}
