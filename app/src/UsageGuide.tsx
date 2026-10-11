import { Camera, Clock3, Film, Info, ScanEye, Upload, X } from "lucide-react";

export default function UsageGuide({ onClose }: { onClose: () => void }) {
  return (
    <aside
      id="usage-guide"
      className="usage-drawer"
      aria-labelledby="guide-title"
    >
      <header className="pane-heading">
        <h2 id="guide-title">Usage guide</h2>
        <button
          className="icon-button"
          aria-label="Close usage guide"
          autoFocus
          onClick={onClose}
        >
          <X size={17} />
        </button>
      </header>
      <div className="guide-content">
        <section aria-labelledby="guide-input">
          <h3 id="guide-input">
            <Upload size={18} aria-hidden="true" />
            <span>1. Prepare your input</span>
          </h3>
          <p>
            Choose <strong>Upload media</strong> or select an example. Examples load
            their input preparation and shot settings. Drag the crop box or its edges
            to frame your image or video, then select <strong>Prepare Cache</strong>.
          </p>
          <p>
            Videos are sampled to <strong>81 frames</strong>. The full clip is
            selected by default. To choose a segment, adjust <strong>Start</strong>{" "}
            and <strong>Sampling FPS</strong>; check <strong>Selected range</strong>{" "}
            before confirming. The selected time span is kept through generation.
          </p>
        </section>

        <section aria-labelledby="guide-controls">
          <h3 id="guide-controls">
            <Camera size={18} aria-hidden="true" />
            <span>2. Set up your shot</span>
          </h3>
          <p>For images, choose a <strong>Conditioning</strong> mode:</p>
          <dl className="guide-modes">
            <div>
              <dt>First-frame only</dt>
              <dd>
                Start a video from the uploaded image. Image features guide the
                first frame; the model generates later frames using the camera
                motion and optional Prompt. <strong>Moving the scene view does
                not reproject the image condition.</strong> Its preview contains
                the image at the start and empty conditions afterwards.
              </dd>
            </div>
            <div>
              <dt>Geometry-guided</dt>
              <dd>
                Choose this to generate from a different viewpoint. The image’s
                reconstructed scene guides every frame along your camera path.
                The condition preview follows the camera while the scene itself
                stays still. <strong>Unseen regions have no cached content</strong>{" "}
                and are filled by the video model. The generated video can add motion.
              </dd>
            </div>
          </dl>
          <dl className="guide-controls">
            <div>
              <dt>Camera</dt>
              <dd>
                Drag the scene to choose a starting view. <strong>Fixed</strong>{" "}
                holds it; the other presets move from it. Videos also offer the
                original camera path and an offset path. <strong>Dolly</strong>{" "}
                moves the camera closer or farther;
                <strong> Zoom</strong> changes the lens.
              </dd>
            </div>
            <div>
              <dt>Time</dt>
              <dd>
                For videos, play normally, freeze a moment, or reverse the action.
                Camera motion continues when scene time is frozen.
              </dd>
            </div>
          </dl>
          <div className="guide-shortcut">
            <span>Move closer / farther</span>
            <span><kbd>⌘</kbd> + scroll on Mac · <kbd>Ctrl</kbd> + scroll on Windows</span>
          </div>
          <p className="guide-caption">
            The colored points show scene shape and motion. The Cache also stores
            the other visual representations used for generation.
          </p>
        </section>

        <section aria-labelledby="guide-conditions">
          <h3 id="guide-conditions">
            <ScanEye size={18} aria-hidden="true" />
            <span>3. Preview conditions</span>
            <em>Optional</em>
          </h3>
          <p>
            Select <strong>Preview conditions</strong> to render the current shot’s
            conditions. <strong>Changing settings does not update these videos
            automatically.</strong> Select <strong>Update preview</strong> after
            changing the camera, Conditioning mode, time or representation set.
          </p>
          <p>
            Use <strong>Shot preview</strong> to play or scrub the scene and
            conditions together. Open a condition card to play it independently.
            Your previous preview remains available until replaced; returning to
            its settings makes it current again.
          </p>
          <p>
            <strong>Mask</strong> shows where the Cache provides content: static
            regions, moving regions, and uncovered areas for the model to fill.
          </p>
          <div className="guide-note">
            <Info size={16} aria-hidden="true" />
            <p>
              Estimated scene shape and motion can contain errors. These can affect
              the mask, conditions and generated result.
            </p>
          </div>
        </section>

        <section aria-labelledby="guide-generate">
          <h3 id="guide-generate">
            <Film size={18} aria-hidden="true" />
            <span>4. Generate & download</span>
          </h3>
          <p>
            Optionally add a <strong>Prompt</strong>, then select{" "}
            <strong>Generate video</strong>.
            You can generate directly without previewing conditions.
          </p>
          <p>
            Open the result to play it at a larger size. <strong>Comparison</strong>{" "}
            → <strong>Prepare comparison</strong> puts the processed input, mask,
            conditions and result side by side, using the settings that generated
            this video.
          </p>
          <ul className="guide-downloads" aria-label="Export choices">
            <li><strong>Generated video</strong><span>The result as an MP4</span></li>
            <li><strong>All videos (.zip)</strong><span>Each video in a separate file</span></li>
            <li><strong>Comparison video</strong><span>The labeled grid as one MP4</span></li>
          </ul>
          <p className="guide-caption">
            Find these under <strong>Export</strong>. Prepare the comparison to
            enable all three choices.
          </p>
        </section>

        <div className="guide-waiting">
          <h3><Clock3 size={16} aria-hidden="true" />About timing</h3>
          <p>
            Timings on Hugging Face’s shared hardware are measured without warmup
            and may be higher than the warmed-up benchmarks reported in the paper.
          </p>
        </div>
        <p className="guide-status">
          This Demo generates one shot at a time. For Rollout, see the{" "}
          <a
            href="https://github.com/ZGCA-HMI-Lab/4LC"
            target="_blank"
            rel="noreferrer"
          >
            code repository
          </a>.
        </p>
      </div>
    </aside>
  );
}
