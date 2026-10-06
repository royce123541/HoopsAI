"use client";

import {
  CartesianGrid,
  ReferenceLine,
  ResponsiveContainer,
  Scatter,
  ScatterChart,
  Tooltip,
  XAxis,
  YAxis,
  type TooltipContentProps,
} from "recharts";

import type { ReliabilityBin } from "@/lib/api";

const pct = (v: number) => `${Math.round(v * 100)}%`;

function BinTooltip({ active, payload }: TooltipContentProps) {
  const b = payload?.[0]?.payload as ReliabilityBin | undefined;
  if (!active || !b) return null;
  return (
    <div className="rounded-md border border-line bg-surface px-3 py-2 text-sm shadow-sm">
      <div className="text-ink-2">
        {b.count.toLocaleString("en-US")} games given {pct(b.bin_low)}–
        {pct(b.bin_high)}
      </div>
      <div className="text-ink">
        Predicted {pct(b.mean_pred)}, home team won{" "}
        <span className="font-semibold">{pct(b.observed)}</span>
      </div>
    </div>
  );
}

/** 12px marker (spec: >= 8px) with a 2px surface ring so it stays legible on the diagonal. */
function BinDot({ cx, cy }: { cx?: number; cy?: number }) {
  if (cx === undefined || cy === undefined) return null;
  return (
    <circle
      cx={cx}
      cy={cy}
      r={6}
      fill="var(--home)"
      stroke="var(--surface)"
      strokeWidth={2}
    />
  );
}

/** Predicted home-win probability vs how often the home team actually won, per bin. Points
 * on the diagonal mean the probabilities can be taken at face value. */
export function CalibrationChart({ bins }: { bins: ReliabilityBin[] }) {
  return (
    // Fixed aspect ratio on the container itself: an aspect-ratio box around a 100%-height
    // ResponsiveContainer feeds its own size back into layout.
    <div className="w-full max-w-md">
      <ResponsiveContainer width="100%" aspect={1}>
        <ScatterChart margin={{ top: 8, right: 16, bottom: 24, left: 8 }}>
          <CartesianGrid stroke="var(--grid)" />
          <XAxis
            type="number"
            dataKey="mean_pred"
            domain={[0, 1]}
            ticks={[0, 0.25, 0.5, 0.75, 1]}
            tickFormatter={pct}
            stroke="var(--axis)"
            tick={{ fill: "var(--muted)", fontSize: 13 }}
            tickLine={false}
            label={{
              value: "Predicted chance home team wins",
              position: "bottom",
              offset: 8,
              fill: "var(--ink-2)",
              fontSize: 13,
            }}
          />
          <YAxis
            type="number"
            dataKey="observed"
            domain={[0, 1]}
            ticks={[0, 0.25, 0.5, 0.75, 1]}
            tickFormatter={pct}
            width={44}
            stroke="var(--axis)"
            tick={{ fill: "var(--muted)", fontSize: 13 }}
            tickLine={false}
            axisLine={false}
          />
          <ReferenceLine
            segment={[
              { x: 0, y: 0 },
              { x: 1, y: 1 },
            ]}
            stroke="var(--axis)"
            label={{
              value: "Perfectly calibrated",
              position: "insideTopLeft",
              fill: "var(--muted)",
              fontSize: 12,
            }}
          />
          <Tooltip
            content={BinTooltip}
            cursor={false}
            isAnimationActive={false}
          />
          <Scatter data={bins} shape={BinDot} isAnimationActive={false} />
        </ScatterChart>
      </ResponsiveContainer>
    </div>
  );
}
