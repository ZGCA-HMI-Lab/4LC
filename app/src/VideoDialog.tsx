import { useEffect, useRef } from "react";
import type { ReactNode } from "react";
import { X } from "lucide-react";

type Props = {
  title: string;
  src: string | null;
  initialTime: number;
  onClose: () => void;
  comparison?: {
    src: string | null;
    stage: string | null;
    error: string | null;
    onPrepare: () => void;
    selected: boolean;
    onSelect: (selected: boolean) => void;
  };
  mask?: boolean;
  exportControl?: ReactNode;
};

export default function VideoDialog({
  title,
  src,
  initialTime,
  onClose,
  comparison,
  mask,
  exportControl,
}: Props) {
  const dialog = useRef<HTMLDialogElement>(null);
  const compare = comparison?.selected ?? false;
  const playbackTime = useRef(initialTime);
  const selected = compare ? comparison!.src : src;
  useEffect(() => {
    dialog.current!.showModal();
  }, []);
  return (
    <dialog
      ref={dialog}
      className="video-dialog"
      aria-label={title}
      onCancel={onClose}
    >
      <header className="pane-heading">
        <h2>{title}</h2>
        {comparison && (
          <div className="segmented" role="group" aria-label="Video view">
            <button
              className={!compare ? "active" : ""}
              aria-pressed={!compare}
              onClick={() => comparison.onSelect(false)}
            >
              Result
            </button>
            <button
              className={compare ? "active" : ""}
              aria-pressed={compare}
              onClick={() => comparison.onSelect(true)}
            >
              Comparison
            </button>
          </div>
        )}
        {exportControl}
        <button
          className="icon-button"
          aria-label="Close video"
          onClick={onClose}
        >
          <X size={18} />
        </button>
      </header>
      <div className="video-dialog-content">
        {selected ? (
          <video
            key={selected}
            src={selected}
            controls
            playsInline
            preload="auto"
            aria-label={compare ? "Comparison video player" : title + " player"}
            onLoadedMetadata={(event) => {
              event.currentTarget.currentTime = Math.min(
                playbackTime.current,
                event.currentTarget.duration,
              );
            }}
            onTimeUpdate={(event) => {
              playbackTime.current = event.currentTarget.currentTime;
            }}
          />
        ) : (
          <div className="comparison-empty">
            {comparison?.stage ? (
              <>
                <span role="status">{comparison.stage}</span>
                <progress aria-label="Comparison progress" />
              </>
            ) : (
              <button
                className="button primary"
                onClick={comparison?.onPrepare}
              >
                Prepare comparison
              </button>
            )}
            {comparison?.error && (
              <p className="notice" role="alert">
                {comparison.error}
              </p>
            )}
          </div>
        )}
      </div>
      {mask && <MaskLegend />}
    </dialog>
  );
}

export function MaskLegend() {
  return (
    <div className="mask-legend" aria-label="Coverage and motion legend">
      <span>
        <i className="uncovered" />
        Uncovered
      </span>
      <span>
        <i className="static" />
        Static
      </span>
      <span>
        <i className="dynamic" />
        Dynamic
      </span>
    </div>
  );
}
