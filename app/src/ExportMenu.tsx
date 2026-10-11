import { useEffect, useRef, useState } from "react";
import { ChevronDown } from "lucide-react";
import type { ComparisonVideo } from "./api";

export default function ExportMenu({
  video,
  comparison,
  onComparison,
  disabled,
}: {
  video: string;
  comparison: ComparisonVideo | null;
  onComparison: () => void;
  disabled: boolean;
}) {
  const root = useRef<HTMLDivElement>(null);
  const [open, setOpen] = useState(false);
  useEffect(() => {
    if (!open) return;
    const outside = (event: PointerEvent) => {
      if (!root.current?.contains(event.target as Node)) setOpen(false);
    };
    const escape = (event: KeyboardEvent) => {
      if (event.key === "Escape") setOpen(false);
    };
    document.addEventListener("pointerdown", outside);
    document.addEventListener("keydown", escape);
    return () => {
      document.removeEventListener("pointerdown", outside);
      document.removeEventListener("keydown", escape);
    };
  }, [open]);
  return (
    <div ref={root} className="export-menu">
      <button
        className="button"
        disabled={disabled}
        aria-expanded={open}
        onClick={() => setOpen(!open)}
      >
        Export <ChevronDown size={14} />
      </button>
      {open && (
        <div className="export-options">
          <a
            href={video}
            download="4lc-video.mp4"
            onClick={() => setOpen(false)}
          >
            Generated video
          </a>
          {comparison ? (
            <>
              <a
                href={comparison.archiveUrl}
                download="4lc-videos.zip"
                onClick={() => setOpen(false)}
              >
                All videos (.zip)
              </a>
              <a
                href={comparison.url}
                download="4lc-comparison.mp4"
                onClick={() => setOpen(false)}
              >
                Comparison video
              </a>
            </>
          ) : (
            <button
              onClick={() => {
                setOpen(false);
                onComparison();
              }}
            >
              Prepare comparison…
            </button>
          )}
        </div>
      )}
    </div>
  );
}
