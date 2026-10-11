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
            Choose <strong>Upload media</strong>. Drag the crop box or its edges
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
          <dl className="guide-controls">
            <div>
              <dt>Camera</dt>
              <dd>
                Choose a camera move, or drag the scene to change the starting
                view. <strong>Dolly</strong> moves the camera closer or farther;
                <strong> Zoom</strong> changes the lens.
              </dd>
            </div>
            <div>
              <dt>Time</dt>
              <dd>
                Play normally, freeze a moment, or reverse the action. Use{" "}
                <strong>Shot preview</strong> to check the camera and time together.
              </dd>
            </div>
          </dl>
          <div className="guide-shortcut">
            <span>Move closer / farther</span>
            <span><kbd>⌘</kbd> + scroll on Mac · <kbd>Ctrl</kbd> + scroll on Windows</span>
          </div>
          <p>
            For images, <strong>Geometry-guided</strong> uses the estimated scene
            shape. <strong>First-frame only</strong> starts the video from your image.
          </p>
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
            Select <strong>Preview conditions</strong> to see what will guide
            generation. The cards follow Shot preview; open any card to play it
            independently.
          </p>
          <p>
            <strong>Mask</strong> shows uncovered, static and moving regions.
            Changing camera or time settings keeps your previous preview. Select{" "}
            <strong>Update preview</strong> to refresh it; returning to the previous
            settings makes it current again.
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
            Write a <strong>Prompt</strong> and select <strong>Generate video</strong>.
            You can generate directly without previewing conditions.
          </p>
          <p>
            Open the result for a larger player. Choose <strong>Comparison</strong>,
            then <strong>Prepare comparison</strong> to view the processed input,
            mask, conditions and result together. It uses the settings from that
            generation, even if you have since changed the controls.
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
          Upload your own media to run the model. This Demo generates one shot at a time. For Rollout and the
          full toolkit, see the{" "}
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
