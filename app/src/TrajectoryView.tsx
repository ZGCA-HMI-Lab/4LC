import { useEffect, useMemo, useRef, useState } from "react";
import { RotateCcw } from "lucide-react";
import { add, cameraFrustum, cameraPose, clamp, dot } from "./camera";
import type { Camera, FrameSize, Vec3 } from "./camera";

type Props = {
  path: Camera[];
  referencePath?: Camera[];
  current: Camera;
  origin: Camera;
  frame: FrameSize;
  disabled: boolean;
};

export default function TrajectoryView({
  path,
  referencePath,
  current,
  origin,
  frame,
  disabled,
}: Props) {
  const ref = useRef<SVGSVGElement>(null);
  const drag = useRef<{
    x: number;
    y: number;
    yaw: number;
    pitch: number;
  } | null>(null);
  const [size, setSize] = useState({ width: 480, height: 350 });
  const [rotation, setRotation] = useState({ yaw: 0.66, pitch: 0.52 });
  useEffect(() => {
    const observer = new ResizeObserver(([entry]) =>
      setSize({
        width: entry.contentRect.width,
        height: entry.contentRect.height,
      }),
    );
    observer.observe(ref.current!);
    return () => observer.disconnect();
  }, []);

  const data = useMemo(() => {
    const base = cameraPose(origin);
    const vector = (v: Vec3): Vec3 => [
      dot(v, base.right),
      dot(v, base.up),
      dot(v, base.forward),
    ];
    const point = (p: Vec3) => vector(add(p, base.position, -1));
    return {
      point,
      positions: path.map((camera) => point(camera.position)),
      reference: referencePath?.map((camera) => point(camera.position)),
    };
  }, [path, referencePath, origin]);
  const positions = [...data.positions, ...(data.reference ?? [])];
  const stationary = data.positions.every(
    (p) => Math.hypot(...add(p, data.positions[0], -1)) < 1e-6,
  );
  const low = [0, 1, 2].map((i) => Math.min(...positions.map((p) => p[i])));
  const high = [0, 1, 2].map((i) => Math.max(...positions.map((p) => p[i])));
  const span =
    Math.max(origin.focusDistance * 0.4, ...high.map((v, i) => v - low[i])) *
    1.22;
  const sightLength = span * (stationary ? 0.55 : 0.22);
  const frustumPoints = path.flatMap((camera) =>
    cameraFrustum(camera, sightLength, frame).map(data.point),
  );
  const geometry = [...positions, ...frustumPoints];
  const geometryLow = [0, 1, 2].map((i) =>
    Math.min(...geometry.map((p) => p[i])),
  );
  const geometryHigh = [0, 1, 2].map((i) =>
    Math.max(...geometry.map((p) => p[i])),
  );
  const center = geometryHigh.map((v, i) => (v + geometryLow[i]) / 2) as Vec3;
  const gridX = Math.max((geometryHigh[0] - geometryLow[0]) / 2, span * 0.12);
  const gridZ = Math.max((geometryHigh[2] - geometryLow[2]) / 2, span * 0.12);
  const floor = geometryLow[1] - span * 0.04;
  const axisOrigin: Vec3 = [center[0] - gridX, floor, center[2] - gridZ];
  const axes = [
    { name: "Right", end: add(axisOrigin, [gridX * 2, 0, 0]) },
    { name: "Forward", end: add(axisOrigin, [0, 0, gridZ * 2]) },
    { name: "Up", end: add(axisOrigin, [0, span * 0.2, 0]) },
  ];
  const rotate = (p: Vec3) => {
    const [x, y, z] = add(p, center, -1);
    return {
      x: x * Math.cos(rotation.yaw) - z * Math.sin(rotation.yaw),
      y:
        -y * Math.cos(rotation.pitch) +
        (x * Math.sin(rotation.yaw) + z * Math.cos(rotation.yaw)) *
          Math.sin(rotation.pitch),
    };
  };
  // Fit the whole shot, including its lens changes, so playback never normalizes
  // a narrowing/widening frustum back to the same apparent size.
  const bounds = [
    ...positions,
    ...frustumPoints,
    ...axes.map((axis) => axis.end),
    ...[-1, 1].flatMap((x) =>
      [-1, 1].map(
        (z) => [center[0] + x * gridX, floor, center[2] + z * gridZ] as Vec3,
      ),
    ),
  ].map(rotate);
  const xMin = Math.min(...bounds.map((p) => p.x)),
    xMax = Math.max(...bounds.map((p) => p.x));
  const yMin = Math.min(...bounds.map((p) => p.y)),
    yMax = Math.max(...bounds.map((p) => p.y));
  // Wide plots share the footer row with the legend (see styles.css).
  const top = size.width < 420 ? 44 : 24;
  const bottom = 52;
  const scale = Math.min(
    (size.width - 56) / (xMax - xMin),
    (size.height - top - bottom) / (yMax - yMin),
  );
  const project = (p: Vec3) => {
    const q = rotate(p);
    return {
      x: size.width / 2 + (q.x - (xMin + xMax) / 2) * scale,
      y:
        top +
        (size.height - top - bottom) / 2 +
        (q.y - (yMin + yMax) / 2) * scale,
    };
  };
  const pointString = (points: Vec3[]) =>
    points
      .map((p) => {
        const q = project(p);
        return q.x + "," + q.y;
      })
      .join(" ");
  const line = (a: Vec3, b: Vec3, color: string, width = 1, dash?: string) => {
    const p = project(a),
      q = project(b);
    return (
      <line
        x1={p.x}
        y1={p.y}
        x2={q.x}
        y2={q.y}
        stroke={color}
        strokeWidth={width}
        strokeDasharray={dash}
      />
    );
  };
  const first = path[0],
    last = path[path.length - 1];
  const sameView =
    stationary &&
    path.every(
      (camera) =>
        Math.hypot(
          ...add(cameraPose(camera).forward, cameraPose(first).forward, -1),
        ) < 1e-6 &&
        Math.hypot(...add(cameraPose(camera).up, cameraPose(first).up, -1)) <
          1e-6 &&
        Math.abs(camera.fov - first.fov) < 1e-6 &&
        camera.focalRatio === first.focalRatio &&
        camera.skewRatio === first.skewRatio &&
        camera.principal.every((v, i) => v === first.principal[i]),
    );
  const start = project(data.positions[0]),
    end = project(data.positions[data.positions.length - 1]);
  const startLabel =
    stationary && !sameView
      ? project(data.point(cameraFrustum(first, sightLength, frame)[0]))
      : start;
  const endLabel = stationary
    ? project(data.point(cameraFrustum(last, sightLength, frame)[2]))
    : end;
  const frustum = (
    camera: Camera,
    color: string,
    width: number,
    dashed = false,
  ) => {
    const apex = data.point(camera.position);
    const corners = cameraFrustum(camera, sightLength, frame).map(data.point);
    return corners.map((p, i) => (
      <g key={i}>
        {line(apex, p, color, width, dashed ? "4 3" : undefined)}
        {line(
          p,
          corners[(i + 1) % 4],
          color,
          width,
          dashed ? "4 3" : undefined,
        )}
      </g>
    ));
  };
  return (
    <div className="trajectory-view">
      <div className="plot-legend">
        {!sameView && (
          <>
            <span>
              <i className="start-dot" />
              Start
            </span>
            <span>
              <i className="end-dot" />
              End
            </span>
          </>
        )}
        <span>
          <i className="current-stroke" />
          Current view
        </span>
        {referencePath && (
          <span>
            <i className="reference-stroke" />
            Input
          </span>
        )}
      </div>
      <svg
        ref={ref}
        aria-label="3D camera trajectory. Drag to rotate."
        role="img"
        onPointerDown={(event) => {
          event.currentTarget.setPointerCapture(event.pointerId);
          drag.current = { x: event.clientX, y: event.clientY, ...rotation };
        }}
        onPointerMove={(event) => {
          if (drag.current)
            setRotation({
              yaw: drag.current.yaw + (event.clientX - drag.current.x) * 0.008,
              pitch: clamp(
                drag.current.pitch + (event.clientY - drag.current.y) * 0.005,
                -0.1,
                1.2,
              ),
            });
        }}
        onPointerUp={() => {
          drag.current = null;
        }}
        onPointerCancel={() => {
          drag.current = null;
        }}
      >
        {!disabled && (
          <>
            {Array.from({ length: 5 }, (_, i) => {
              const x = -gridX + (i * gridX) / 2;
              const z = -gridZ + (i * gridZ) / 2;
              return (
                <g key={i}>
                  {line(
                    [center[0] + x, floor, center[2] - gridZ],
                    [center[0] + x, floor, center[2] + gridZ],
                    "#dce2e5",
                  )}
                  {line(
                    [center[0] - gridX, floor, center[2] + z],
                    [center[0] + gridX, floor, center[2] + z],
                    "#dce2e5",
                  )}
                </g>
              );
            })}
            {axes.map((axis) => {
              const p = project(axis.end);
              return (
                <g key={axis.name}>
                  {line(axisOrigin, axis.end, "#b5c0c5")}
                  <text x={p.x + 5} y={p.y + 14} fill="#65717b" fontSize="11">
                    {axis.name}
                  </text>
                </g>
              );
            })}
            {data.reference && (
              <polyline
                data-path="reference"
                points={pointString(data.reference)}
                fill="none"
                stroke="#a8b4ba"
                strokeWidth="1.5"
                strokeDasharray="4 4"
              />
            )}
            <polyline
              data-path="target"
              points={pointString(data.positions)}
              fill="none"
              stroke="#526776"
              strokeWidth="2.5"
              strokeLinejoin="round"
            />
            {!stationary &&
              path
                .filter((_, i) => i % 5 === 0)
                .map((camera, i) => {
                  const pose = cameraPose(camera),
                    a = project(data.point(camera.position));
                  const b = project(
                    data.point(
                      add(camera.position, pose.forward, sightLength * 0.6),
                    ),
                  );
                  const angle = Math.atan2(b.y - a.y, b.x - a.x),
                    head = 4;
                  return (
                    <g key={i} stroke="#839aab" fill="none" strokeWidth="1">
                      <line x1={a.x} y1={a.y} x2={b.x} y2={b.y} />
                      <path
                        d={`M${b.x - Math.cos(angle - 0.45) * head},${b.y - Math.sin(angle - 0.45) * head} L${b.x},${b.y} L${b.x - Math.cos(angle + 0.45) * head},${b.y - Math.sin(angle + 0.45) * head}`}
                      />
                    </g>
                  );
                })}
            {!sameView && (
              <>
                <g data-camera="start">{frustum(first, "#4a9067", 1, true)}</g>
                <g data-camera="end">{frustum(last, "#be7965", 1, true)}</g>
              </>
            )}
            <circle
              cx={start.x}
              cy={start.y}
              r="4"
              fill="#3d9b66"
              stroke="#fff"
              strokeWidth="1.5"
            />
            {!stationary && (
              <circle
                cx={end.x}
                cy={end.y}
                r="4"
                fill="#cd705e"
                stroke="#fff"
                strokeWidth="1.5"
              />
            )}
            <text
              x={startLabel.x + 8}
              y={startLabel.y - 8}
              fontSize="11"
              fill="#386b4b"
            >
              {sameView ? "Fixed" : "Start"}
            </text>
            {!sameView && (
              <text
                x={endLabel.x + 8}
                y={endLabel.y - 8}
                fontSize="11"
                fill="#985441"
              >
                End
              </text>
            )}
            <g
              data-camera="current"
              data-fov={current.fov}
              data-position={current.position.join(",")}
            >
              {frustum(current, "#253f4c", 1.7)}
            </g>
          </>
        )}
      </svg>
      <div className="plot-footer">
        <div className="plot-camera-info">
          <span>{((current.fov * 180) / Math.PI).toFixed(1)}° FOV</span>
          <button
            className="icon-button"
            aria-label="Reset trajectory view"
            onClick={() => setRotation({ yaw: 0.66, pitch: 0.52 })}
          >
            <RotateCcw size={16} />
          </button>
        </div>
      </div>
    </div>
  );
}
