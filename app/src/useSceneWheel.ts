import { useEffect, useRef } from "react";
import type { RefObject } from "react";
import { dollyCamera } from "./camera";
import type { Camera } from "./camera";
import { wheelDollyDistance } from "./wheel.ts";

export default function useSceneWheel(
  ref: RefObject<HTMLElement | null>,
  camera: Camera,
  onChange: ((camera: Camera) => void) | undefined,
) {
  const latest = useRef({ camera, onChange });
  latest.current = { camera, onChange };
  useEffect(() => {
    const element = ref.current!;
    const mac = /Mac|iPhone|iPad/.test(navigator.platform);
    const wheel = (event: WheelEvent) => {
      if (!(mac ? event.metaKey : event.ctrlKey) || !latest.current.onChange)
        return;
      event.preventDefault();
      const distance = wheelDollyDistance(
        latest.current.camera.focusDistance,
        event.deltaY,
        event.deltaMode,
        element.clientHeight,
      );
      const next = dollyCamera(latest.current.camera, distance);
      latest.current.camera = next;
      latest.current.onChange(next);
    };
    element.addEventListener("wheel", wheel, { passive: false });
    return () => element.removeEventListener("wheel", wheel);
  }, [ref]);
}
