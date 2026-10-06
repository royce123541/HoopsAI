"use client";

import {
  CartesianGrid,
  Line,
  LineChart,
  ReferenceLine,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
  type TooltipContentProps,
} from "recharts";

import type { TeamRef, WinProbPoint } from "@/lib/api";
import { gameClock } from "@/lib/format";

const REGULATION = 2880;
const OT = 300;

/** Probability of the team ahead in the estimate, e.g. 0.3 home -> "PHI 70%". */
function leaderLabel(p: number, home: TeamRef, away: TeamRef): string {
  if (Math.abs(p - 0.5) < 0.005) return "50%";
  return p > 0.5
    ? `${home.abbreviation} ${Math.round(p * 100)}%`
    : `${away.abbreviation} ${Math.round((1 - p) * 100)}%`;
}

function quarterTicks(end: number): number[] {
  const ticks = [0, 720, 1440, 2160, 2880];
  for (let t = REGULATION + OT; t <= end; t += OT) ticks.push(t);
  return ticks;
}

function tickLabel(t: number): string {
  if (t === 0) return "Tip";
  if (t <= REGULATION) return `Q${t / 720} end`;
  const ot = (t - REGULATION) / OT;
  return ot === 1 ? "OT end" : `${ot}OT end`;
}

/**
 * Home team's chance to win across the game. The y-axis is labelled by team, so the top half
 * reads as "home ahead" and the bottom as "away ahead"; the 50% line is the even mark.
 */
export function WinProbChart({
  points,
  home,
  away,
}: {
  points: WinProbPoint[];
  home: TeamRef;
  away: TeamRef;
}) {
  const end = Math.max(REGULATION, ...points.map((p) => p.elapsed_seconds));
  const xMax =
    end <= REGULATION
      ? REGULATION
      : REGULATION + Math.ceil((end - REGULATION) / OT) * OT;

  function PointTooltip({ active, payload }: TooltipContentProps) {
    const p = payload?.[0]?.payload as WinProbPoint | undefined;
    if (!active || !p) return null;
    return (
      <div className="max-w-64 rounded-md border border-line bg-surface px-3 py-2 text-sm shadow-sm">
        <div className="text-ink-2">
          {p.action_id === 0 ? "Tip-off" : gameClock(p.period, p.clock_seconds)}
          ,{" "}
          <span className="tabular">
            {away.abbreviation} {p.score_away}, {home.abbreviation}{" "}
            {p.score_home}
          </span>
        </div>
        <div className="font-semibold text-ink">
          {leaderLabel(p.home_win_prob, home, away)} to win
        </div>
        {p.description && p.action_id !== 0 && (
          <div className="mt-0.5 text-ink-2">{p.description}</div>
        )}
      </div>
    );
  }

  return (
    <div className="h-72 w-full">
      <ResponsiveContainer width="100%" height="100%">
        <LineChart
          data={points}
          margin={{ top: 8, right: 16, bottom: 4, left: 8 }}
        >
          <CartesianGrid vertical={false} stroke="var(--grid)" />
          <XAxis
            dataKey="elapsed_seconds"
            type="number"
            domain={[0, xMax]}
            ticks={quarterTicks(xMax)}
            tickFormatter={tickLabel}
            stroke="var(--axis)"
            tick={{ fill: "var(--muted)", fontSize: 13 }}
            tickLine={false}
          />
          <YAxis
            domain={[0, 1]}
            ticks={[0, 0.25, 0.5, 0.75, 1]}
            tickFormatter={(v: number) => leaderLabel(v, home, away)}
            width={72}
            stroke="var(--axis)"
            tick={{ fill: "var(--muted)", fontSize: 13 }}
            tickLine={false}
            axisLine={false}
          />
          <ReferenceLine y={0.5} stroke="var(--ink-2)" />
          <Tooltip
            content={PointTooltip}
            cursor={{ stroke: "var(--axis)" }}
            isAnimationActive={false}
          />
          <Line
            dataKey="home_win_prob"
            type="stepAfter"
            stroke="var(--home)"
            strokeWidth={2}
            strokeLinejoin="round"
            dot={false}
            activeDot={{
              r: 5,
              fill: "var(--home)",
              stroke: "var(--surface)",
              strokeWidth: 2,
            }}
            isAnimationActive={false}
          />
        </LineChart>
      </ResponsiveContainer>
    </div>
  );
}
