import { useRef } from "react";
import { Upload } from "lucide-react";
import type { Example } from "./api";
export type UploadedMedia = {
  name: string;
  url: string;
  kind: "video" | "image";
  exampleId?: string;
};
type Props = {
  examples: Example[];
  uploaded: UploadedMedia | null;
  onExample: (example: Example) => void;
  onUpload: (file: File) => void;
  onPreview: () => void;
  onPrepare: () => void;
  preparing: boolean;
  busy: boolean;
  prepared: boolean;
};

export default function InputPanel({
  examples,
  uploaded,
  onExample,
  onUpload,
  onPreview,
  onPrepare,
  preparing,
  busy,
  prepared,
}: Props) {
  const uploadRef = useRef<HTMLInputElement>(null);

  return (
    <section className="input-panel" aria-label="Input source">
      <header className="input-heading">
        <h2>Input</h2>
      </header>
      <div className="input-upload">
        <button className="button" onClick={() => uploadRef.current?.click()}>
          <Upload size={14} />
          Upload media
        </button>
        {uploaded && (
          <div className="input-example selected">
            <button
              className="input-choice"
              aria-label="Preview input"
              onClick={onPreview}
            >
              <span className="input-thumbnail">
                {uploaded.kind === "image" ? (
                  <img src={uploaded.url} alt="Uploaded input" />
                ) : (
                  <video src={uploaded.url} muted playsInline preload="auto" />
                )}
              </span>
              <span className="input-description">
                <strong>{uploaded.name}</strong>
                <span>
                  Uploaded · {uploaded.kind === "image" ? "Image" : "Video"}
                </span>
              </span>
            </button>
          </div>
        )}
        {uploaded && (
          <button
            className="button primary"
            onClick={onPrepare}
            disabled={preparing || busy}
          >
            {preparing
              ? "Preparing…"
              : prepared
                ? "Rebuild Cache"
                : "Prepare Cache"}
          </button>
        )}
      </div>
      {examples.length > 0 && <h3 className="examples-heading">Examples</h3>}
      <div className="input-list">
        {examples.map((item) => {
          const selected = uploaded?.exampleId === item.id;
          return (
            <div
              className={"input-example" + (selected ? " selected" : "")}
              key={item.id}
            >
              <button
                className="input-choice"
                aria-label={"Load " + item.name}
                aria-pressed={selected}
                disabled={busy || preparing}
                onClick={() => onExample(item)}
              >
                <span className="input-thumbnail">
                  {item.kind === "image" ? (
                    <img src={item.url} alt={item.name} />
                  ) : (
                    <video
                      src={item.url}
                      muted
                      playsInline
                      preload="metadata"
                    />
                  )}
                </span>
                <span className="input-description">
                  <strong>{item.name}</strong>
                  <span>{item.kind === "image" ? "Image" : "Video"}</span>
                </span>
              </button>
            </div>
          );
        })}
      </div>
      <input
        ref={uploadRef}
        className="sr-only"
        type="file"
        accept="image/*,video/*"
        onChange={(event) => {
          const file = event.target.files?.[0];
          if (file) onUpload(file);
          event.target.value = "";
        }}
      />
    </section>
  );
}
