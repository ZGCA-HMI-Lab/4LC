import useSceneWheel from "./useSceneWheel";
import { useEffect, useRef } from "react";
import {
  BufferAttribute,
  BufferGeometry,
  Camera as RenderCamera,
  Points,
  Scene,
  ShaderMaterial,
  Vector3,
  Vector4,
  WebGLRenderer,
} from "three";
import { cameraIntrinsics, cameraPose, orbitCamera } from "./camera";
import type { Camera, FrameSize } from "./camera";
import { sourceBracket } from "./scene";
import type { SceneData } from "./scene";

type Props = {
  data: SceneData;
  camera: Camera;
  frame: FrameSize;
  time: number;
  onCameraChange: (camera: Camera) => void;
  onError: (message: string) => void;
};

export default function PointCloudScene(props: Props) {
  const canvas = useRef<HTMLCanvasElement>(null);
  useSceneWheel(canvas, props.camera, props.onCameraChange);
  const latest = useRef(props);
  latest.current = props;
  const draw = useRef<(() => void) | null>(null);
  const drag = useRef<{ x: number; y: number; camera: Camera } | null>(null);

  useEffect(() => {
    const element = canvas.current!;
    let renderer: WebGLRenderer;
    try {
      renderer = new WebGLRenderer({ canvas: element, antialias: false });
    } catch {
      latest.current.onError(
        "This browser could not initialize WebGL2 for the scene.",
      );
      return;
    }
    renderer.setClearColor("#182127");
    const world = new Scene();
    const dummyCamera = new RenderCamera();
    const { data } = latest.current;
    const uniforms = {
      cameraPositionWorld: { value: new Vector3() },
      cameraRight: { value: new Vector3() },
      cameraUp: { value: new Vector3() },
      cameraForward: { value: new Vector3() },
      lens: { value: new Vector4() },
      skew: { value: 0 },
      clock: { value: 0 },
      leftSource: { value: 0 },
      rightSource: { value: 0 },
      sourceTimes: { value: new Float32Array(data.sourceTimes) },
      pointSize: { value: 1 },
      nearPlane: { value: data.focusDistance / 10000 },
      farPlane: { value: data.radius * 10 },
    };
    const geometries: BufferGeometry[] = [];
    const materials: ShaderMaterial[] = [];
    for (const moving of [false, true]) {
      const cloud = moving ? data.dynamic : data.static;
      if (!cloud.positions.length) continue;
      const geometry = new BufferGeometry();
      geometry.setAttribute(
        "position",
        new BufferAttribute(cloud.positions, 3),
      );
      geometry.setAttribute("rgb", new BufferAttribute(cloud.colors, 3, true));
      if (moving) {
        geometry.setAttribute(
          "velocityFwd",
          new BufferAttribute(data.dynamic.forward, 3),
        );
        geometry.setAttribute(
          "velocityBwd",
          new BufferAttribute(data.dynamic.backward, 3),
        );
        geometry.setAttribute(
          "source",
          new BufferAttribute(data.dynamic.sources, 1),
        );
      }
      const material = new ShaderMaterial({
        uniforms,
        defines: moving ? { MOVING: 1 } : {},
        vertexShader: `
          attribute vec3 rgb;
          varying vec3 pointColor;
          uniform vec3 cameraPositionWorld, cameraRight, cameraUp, cameraForward;
          uniform vec4 lens;
          uniform float skew, clock, pointSize, nearPlane, farPlane;
          #ifdef MOVING
          attribute vec3 velocityFwd, velocityBwd;
          attribute float source;
          uniform int leftSource, rightSource;
          uniform float sourceTimes[${data.sourceTimes.length}];
          #endif
          void main() {
            vec3 point = position;
            #ifdef MOVING
            int index = int(source);
            if (index != leftSource && index != rightSource) {
              gl_Position = vec4(0.0, 0.0, 2.0, 1.0);
              gl_PointSize = 1.0;
              return;
            }
            float dt = clock - sourceTimes[index];
            point += (dt >= 0.0 ? velocityFwd : velocityBwd) * abs(dt);
            #endif
            point *= vec3(1.0, -1.0, -1.0);
            vec3 relative = point - cameraPositionWorld;
            float x = dot(relative, cameraRight);
            float y = dot(relative, cameraUp);
            float z = dot(relative, cameraForward);
            gl_Position = vec4(
              2.0 * lens.x * x - 2.0 * skew * y + (2.0 * lens.z - 1.0) * z,
              2.0 * lens.y * y + (1.0 - 2.0 * lens.w) * z,
              (farPlane + nearPlane) / (farPlane - nearPlane) * z - 2.0 * farPlane * nearPlane / (farPlane - nearPlane), z);
            gl_PointSize = pointSize;
            pointColor = rgb;
          }
        `,
        fragmentShader: `varying vec3 pointColor; void main() { gl_FragColor = vec4(pointColor, 1.0); }`,
      });
      const points = new Points(geometry, material);
      points.frustumCulled = false;
      world.add(points);
      geometries.push(geometry);
      materials.push(material);
    }
    const render = () => {
      const { camera, frame, time } = latest.current;
      const width = element.clientWidth,
        height = element.clientHeight;
      if (!width || !height) return;
      const ratio = Math.min(window.devicePixelRatio, 2);
      renderer.setPixelRatio(ratio);
      renderer.setSize(width, height, false);
      const scale = Math.min(width / frame.width, height / frame.height);
      renderer.setViewport(
        (width - frame.width * scale) / 2,
        (height - frame.height * scale) / 2,
        frame.width * scale,
        frame.height * scale,
      );
      const pose = cameraPose(camera),
        k = cameraIntrinsics(camera, frame);
      uniforms.cameraPositionWorld.value.fromArray(camera.position);
      uniforms.cameraRight.value.fromArray(pose.right);
      uniforms.cameraUp.value.fromArray(pose.up);
      uniforms.cameraForward.value.fromArray(pose.forward);
      uniforms.lens.value.set(k.fx, k.fy, k.cx, k.cy);
      uniforms.skew.value = k.skew;
      uniforms.clock.value = Math.fround(time);
      [uniforms.leftSource.value, uniforms.rightSource.value] = sourceBracket(
        data.sourceTimes,
        time,
      );
      uniforms.farPlane.value = Math.max(
        data.radius * 10,
        Math.hypot(...camera.position) * 4,
        data.focusDistance * 10,
      );
      uniforms.pointSize.value = 1.25 * scale * ratio;
      renderer.render(world, dummyCamera);
    };
    draw.current = render;
    const observer = new ResizeObserver(render);
    observer.observe(element);
    const lost = (event: Event) => {
      event.preventDefault();
      latest.current.onError(
        "The scene's WebGL context was lost. Reload the page to rebuild it.",
      );
    };
    element.addEventListener("webglcontextlost", lost);
    render();
    return () => {
      draw.current = null;
      observer.disconnect();
      element.removeEventListener("webglcontextlost", lost);
      geometries.forEach((geometry) => geometry.dispose());
      materials.forEach((material) => material.dispose());
      renderer.dispose();
    };
  }, [props.data]);

  useEffect(() => {
    draw.current?.();
  }, [props.camera, props.frame, props.time]);

  return (
    <canvas
      ref={canvas}
      className="scene-canvas interactive"
      role="img"
      aria-label="4D Scene point cloud"
      onPointerDown={(event) => {
        event.currentTarget.setPointerCapture(event.pointerId);
        drag.current = {
          x: event.clientX,
          y: event.clientY,
          camera: props.camera,
        };
      }}
      onPointerMove={(event) => {
        if (drag.current)
          props.onCameraChange(
            orbitCamera(
              drag.current.camera,
              (event.clientX - drag.current.x) * 0.006,
              (event.clientY - drag.current.y) * 0.004,
            ),
          );
      }}
      onPointerUp={() => {
        drag.current = null;
      }}
      onPointerCancel={() => {
        drag.current = null;
      }}
    />
  );
}
