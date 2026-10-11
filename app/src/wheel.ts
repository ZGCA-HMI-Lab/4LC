export function wheelDollyDistance(
  focusDistance: number,
  delta: number,
  mode: number,
  pageHeight: number,
) {
  const pixels = delta * (mode === 1 ? 16 : mode === 2 ? pageHeight : 1);
  return focusDistance * (1 - Math.exp(pixels * 0.0015));
}
