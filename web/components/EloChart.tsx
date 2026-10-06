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

import type { EloPoint } from "@/lib/api";
import { seasonLabel, shortDate } from "@/lib/format";

const LEAGUE_AVERAGE = 1505; // Elo's between-seasons regression target

type Datum = { t: number; elo: number | null; date: string; season: number };

function toData(points: EloPoint[]): Datum[] {
  const data: Datum[] = [];
  points.forEach((p, i) => {
    const prev = points[i - 1];
    if (prev && prev.season !== p.season) {
      // Break the line over the off-season: ratings jump there (regression to the mean).
      data.push({
        t: Date.parse(prev.game_date) + 86_400_000,
        elo: null,
        date: "",
        season: p.season,
      });
    }
    data.push({
      t: Date.parse(p.game_date),
      elo: p.elo,
      date: p.game_date,
      season: p.season,
    });
  });
  return data;
}

function EloTooltip({ active, payload }: TooltipContentProps) {
  const d = payload?.[0]?.payload as Datum | undefined;
  if (!active || !d || d.elo === null) return null;
  return (
    <div className="rounded-md border border-line bg-surface px-3 py-2 text-sm shadow-sm">
      <div className="text-ink-2">{shortDate(d.date, true)}</div>
      <div className="tabular font-semibold text-ink">
        {Math.round(d.elo)} Elo
      </div>
    </div>
  );
}

/** A team's Elo after each game over recent seasons; single series, so no legend box. */
export function EloChart({ points }: { points: EloPoint[] }) {
  const data = toData(points);
  const seasonStarts = points.filter(
    (p, i) => i === 0 || points[i - 1].season !== p.season,
  );
  const elos = points.map((p) => p.elo);
  const lo = Math.floor((Math.min(...elos, LEAGUE_AVERAGE) - 25) / 50) * 50;
  const hi = Math.ceil((Math.max(...elos, LEAGUE_AVERAGE) + 25) / 50) * 50;

  return (
    <div className="h-72 w-full">
      <ResponsiveContainer width="100%" height="100%">
        <LineChart
          data={data}
          margin={{ top: 8, right: 16, bottom: 4, left: 0 }}
        >
          <CartesianGrid vertical={false} stroke="var(--grid)" />
          <XAxis
            dataKey="t"
            type="number"
            scale="time"
            domain={["dataMin", "dataMax"]}
            ticks={seasonStarts.map((p) => Date.parse(p.game_date))}
            tickFormatter={(t: number) => {
              const p = seasonStarts.find((s) => Date.parse(s.game_date) === t);
              return p ? seasonLabel(p.season) : "";
            }}
            stroke="var(--axis)"
            tick={{ fill: "var(--muted)", fontSize: 13 }}
            tickLine={false}
          />
          <YAxis
            domain={[lo, hi]}
            tickCount={5}
            width={44}
            stroke="var(--axis)"
            tick={{ fill: "var(--muted)", fontSize: 13 }}
            tickLine={false}
            axisLine={false}
            className="tabular"
          />
          <ReferenceLine
            y={LEAGUE_AVERAGE}
            stroke="var(--axis)"
            label={{
              value: "League average",
              position: "insideTopLeft",
              fill: "var(--muted)",
              fontSize: 12,
            }}
          />
          <Tooltip
            content={EloTooltip}
            cursor={{ stroke: "var(--axis)" }}
            isAnimationActive={false}
          />
          <Line
            dataKey="elo"
            stroke="var(--home)"
            strokeWidth={2}
            strokeLinejoin="round"
            strokeLinecap="round"
            dot={false}
            activeDot={{
              r: 5,
              fill: "var(--home)",
              stroke: "var(--surface)",
              strokeWidth: 2,
            }}
            connectNulls={false}
            isAnimationActive={false}
          />
        </LineChart>
      </ResponsiveContainer>
    </div>
  );
}
