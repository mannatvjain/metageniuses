import { useRef, useEffect, useState, useCallback } from "react";

/**
 * Canvas-based scatter plot for large point clouds.
 * Much faster than Recharts ScatterChart for 1000+ points.
 *
 * Props:
 *   data: [{ x, y, ...rest }]  — points to render
 *   series: [{ filter, color, opacity, radius, label }]  — render layers
 *   width/height: number
 *   xLabel/yLabel: string
 *   xDomain/yDomain: [min, max] (auto if omitted)
 *   tooltipContent: (point) => string|ReactNode
 *   onPointClick: (point) => void
 */
export default function CanvasScatter({
  data,
  series,
  width: containerWidth,
  height = 400,
  xLabel = "",
  yLabel = "",
  xDomain,
  yDomain,
  tooltipContent,
  onPointClick,
  legend,
}) {
  const canvasRef = useRef(null);
  const containerRef = useRef(null);
  const [tooltip, setTooltip] = useState(null);
  const [dims, setDims] = useState({ w: containerWidth || 600, h: height });

  // Responsive width
  useEffect(() => {
    if (containerWidth) return;
    const el = containerRef.current;
    if (!el) return;
    const ro = new ResizeObserver((entries) => {
      const { width } = entries[0].contentRect;
      if (width > 0) setDims((d) => ({ ...d, w: width }));
    });
    ro.observe(el);
    return () => ro.disconnect();
  }, [containerWidth]);

  const margin = { top: 15, right: 20, bottom: 40, left: 55 };
  const plotW = dims.w - margin.left - margin.right;
  const plotH = dims.h - margin.top - margin.bottom;

  // Compute domains
  const xs = data.map((d) => d.x);
  const ys = data.map((d) => d.y);
  const xMin = xDomain ? xDomain[0] : Math.min(...xs);
  const xMax = xDomain ? xDomain[1] : Math.max(...xs);
  const yMin = yDomain ? yDomain[0] : Math.min(...ys);
  const yMax = yDomain ? yDomain[1] : Math.max(...ys);

  const toCanvasX = useCallback((v) => margin.left + ((v - xMin) / (xMax - xMin)) * plotW, [xMin, xMax, plotW]);
  const toCanvasY = useCallback((v) => margin.top + ((yMax - v) / (yMax - yMin)) * plotH, [yMin, yMax, plotH]);

  // Draw
  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext("2d");
    const dpr = window.devicePixelRatio || 1;
    canvas.width = dims.w * dpr;
    canvas.height = dims.h * dpr;
    ctx.scale(dpr, dpr);
    ctx.clearRect(0, 0, dims.w, dims.h);

    // Background
    ctx.fillStyle = "#fff";
    ctx.fillRect(margin.left, margin.top, plotW, plotH);

    // Grid
    ctx.strokeStyle = "#f0f0f0";
    ctx.lineWidth = 1;
    const xTicks = niceTicksFor(xMin, xMax, 8);
    const yTicks = niceTicksFor(yMin, yMax, 8);
    for (const t of xTicks) {
      const x = toCanvasX(t);
      ctx.beginPath(); ctx.moveTo(x, margin.top); ctx.lineTo(x, margin.top + plotH); ctx.stroke();
    }
    for (const t of yTicks) {
      const y = toCanvasY(t);
      ctx.beginPath(); ctx.moveTo(margin.left, y); ctx.lineTo(margin.left + plotW, y); ctx.stroke();
    }

    // Axes
    ctx.strokeStyle = "#ccc";
    ctx.lineWidth = 1;
    ctx.strokeRect(margin.left, margin.top, plotW, plotH);

    // Tick labels
    ctx.fillStyle = "#828282";
    ctx.font = "11px sans-serif";
    ctx.textAlign = "center";
    for (const t of xTicks) {
      ctx.fillText(t.toFixed(t % 1 ? 1 : 0), toCanvasX(t), margin.top + plotH + 16);
    }
    ctx.textAlign = "right";
    for (const t of yTicks) {
      ctx.fillText(t.toFixed(t % 1 ? 1 : 0), margin.left - 6, toCanvasY(t) + 4);
    }

    // Axis labels
    ctx.fillStyle = "#666";
    ctx.font = "12px sans-serif";
    ctx.textAlign = "center";
    ctx.fillText(xLabel, margin.left + plotW / 2, dims.h - 4);
    ctx.save();
    ctx.translate(14, margin.top + plotH / 2);
    ctx.rotate(-Math.PI / 2);
    ctx.fillText(yLabel, 0, 0);
    ctx.restore();

    // Points — render each series layer
    for (const s of series) {
      const pts = s.filter ? data.filter(s.filter) : data;
      ctx.globalAlpha = s.opacity ?? 0.5;
      ctx.fillStyle = s.color;
      const r = s.radius ?? 2;
      for (const pt of pts) {
        const cx = toCanvasX(pt.x);
        const cy = toCanvasY(pt.y);
        if (cx < margin.left || cx > margin.left + plotW) continue;
        if (cy < margin.top || cy > margin.top + plotH) continue;
        ctx.beginPath();
        ctx.arc(cx, cy, r, 0, Math.PI * 2);
        ctx.fill();
      }
    }
    ctx.globalAlpha = 1;
  }, [data, series, dims, xMin, xMax, yMin, yMax, toCanvasX, toCanvasY, plotW, plotH, xLabel, yLabel]);

  // Mouse interactions
  const handleMouse = useCallback((e) => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const rect = canvas.getBoundingClientRect();
    const mx = e.clientX - rect.left;
    const my = e.clientY - rect.top;

    // Find nearest point
    let best = null;
    let bestDist = 100; // 10px threshold
    for (const pt of data) {
      const cx = toCanvasX(pt.x);
      const cy = toCanvasY(pt.y);
      const dist = Math.sqrt((mx - cx) ** 2 + (my - cy) ** 2);
      if (dist < bestDist) {
        bestDist = dist;
        best = pt;
      }
    }

    if (best && tooltipContent) {
      setTooltip({
        x: e.clientX - rect.left,
        y: e.clientY - rect.top,
        content: tooltipContent(best),
      });
    } else {
      setTooltip(null);
    }
  }, [data, toCanvasX, toCanvasY, tooltipContent]);

  const handleClick = useCallback((e) => {
    if (!onPointClick) return;
    const canvas = canvasRef.current;
    if (!canvas) return;
    const rect = canvas.getBoundingClientRect();
    const mx = e.clientX - rect.left;
    const my = e.clientY - rect.top;

    let best = null;
    let bestDist = 100;
    for (const pt of data) {
      const cx = toCanvasX(pt.x);
      const cy = toCanvasY(pt.y);
      const dist = Math.sqrt((mx - cx) ** 2 + (my - cy) ** 2);
      if (dist < bestDist) { bestDist = dist; best = pt; }
    }
    if (best) onPointClick(best);
  }, [data, toCanvasX, toCanvasY, onPointClick]);

  return (
    <div ref={containerRef} className="relative" style={{ width: containerWidth || "100%" }}>
      <canvas
        ref={canvasRef}
        style={{ width: dims.w, height: dims.h, cursor: onPointClick ? "crosshair" : "default" }}
        onMouseMove={handleMouse}
        onMouseLeave={() => setTooltip(null)}
        onClick={handleClick}
      />
      {tooltip && (
        <div
          className="absolute bg-white border border-gray-200 rounded-md shadow-sm p-2 text-xs pointer-events-none z-10"
          style={{ left: tooltip.x + 12, top: tooltip.y - 10 }}
        >
          {tooltip.content}
        </div>
      )}
      {legend && (
        <div className="flex items-center justify-center gap-6 mt-2 text-xs text-[#828282]">
          {legend.map((l, i) => (
            <span key={i} className="flex items-center gap-1">
              <span className="w-3 h-3 rounded-full inline-block" style={{ backgroundColor: l.color }} />
              {l.label}
            </span>
          ))}
        </div>
      )}
    </div>
  );
}

function niceTicksFor(min, max, count) {
  const range = max - min;
  const rough = range / count;
  const mag = Math.pow(10, Math.floor(Math.log10(rough)));
  const residual = rough / mag;
  let step;
  if (residual <= 1.5) step = 1 * mag;
  else if (residual <= 3.5) step = 2 * mag;
  else if (residual <= 7.5) step = 5 * mag;
  else step = 10 * mag;

  const ticks = [];
  let t = Math.ceil(min / step) * step;
  while (t <= max) {
    ticks.push(Math.round(t * 1e10) / 1e10);
    t += step;
  }
  return ticks;
}
