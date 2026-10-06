import type { SeasonMetrics } from "@/lib/api";
import { percent, seasonLabel } from "@/lib/format";

/**
 * Dumbbell per season: log loss of the model (blue) and the Elo baseline (grey) on one shared
 * scale; lower is better. The right column states the gap, the number the chart is about.
 */
export function SeasonComparison({ seasons }: { seasons: SeasonMetrics[] }) {
  const values = seasons.flatMap((s) => [s.model.logloss, s.elo.logloss]);
  const lo = Math.min(...values) - 0.005;
  const hi = Math.max(...values) + 0.005;
  const x = (v: number) => `${((v - lo) / (hi - lo)) * 100}%`;

  return (
    <div>
      <div className="mb-3 flex gap-5 text-sm text-ink-2">
        <span className="flex items-center gap-2">
          <span aria-hidden className="h-2.5 w-2.5 rounded-full bg-home" />{" "}
          HoopsAI model
        </span>
        <span className="flex items-center gap-2">
          <span aria-hidden className="h-2.5 w-2.5 rounded-full bg-baseline" />{" "}
          Elo baseline
        </span>
        <span className="ml-auto">Log loss, lower is better</span>
      </div>
      <ul className="divide-y divide-line border-y border-line">
        {seasons.map((s) => {
          const gap = s.elo.logloss - s.model.logloss;
          const [left, right] = [
            Math.min(s.model.logloss, s.elo.logloss),
            Math.max(s.model.logloss, s.elo.logloss),
          ];
          return (
            <li
              key={s.season}
              className="grid grid-cols-[4.5rem_1fr_7.5rem] items-center gap-4 py-3"
            >
              <span className="tabular text-[15px] text-ink-2">
                {seasonLabel(s.season)}
              </span>
              <span className="relative block h-3" aria-hidden>
                <span
                  className="absolute top-1/2 h-px -translate-y-1/2 bg-axis"
                  style={{
                    left: x(left),
                    width: `calc(${x(right)} - ${x(left)})`,
                  }}
                />
                <span
                  title={`Elo baseline ${s.elo.logloss.toFixed(4)}`}
                  className="absolute top-1/2 h-3 w-3 -translate-x-1/2 -translate-y-1/2 rounded-full bg-baseline ring-2 ring-[var(--page)]"
                  style={{ left: x(s.elo.logloss) }}
                />
                <span
                  title={`Model ${s.model.logloss.toFixed(4)}`}
                  className="absolute top-1/2 h-3 w-3 -translate-x-1/2 -translate-y-1/2 rounded-full bg-home ring-2 ring-[var(--page)]"
                  style={{ left: x(s.model.logloss) }}
                />
              </span>
              <span className="tabular text-right text-[15px]">
                {gap > 0
                  ? `${gap.toFixed(4)} better`
                  : `${(-gap).toFixed(4)} worse`}
              </span>
            </li>
          );
        })}
      </ul>
      <details className="mt-3 text-sm">
        <summary className="cursor-pointer text-ink-2">Show as a table</summary>
        <table className="tabular mt-2 w-full">
          <thead>
            <tr className="text-left text-ink-2">
              <th className="py-1 font-medium">Season</th>
              <th className="py-1 text-right font-medium">Log loss (Elo)</th>
              <th className="py-1 text-right font-medium">Brier (Elo)</th>
              <th className="py-1 text-right font-medium">Accuracy (Elo)</th>
              <th className="py-1 text-right font-medium">Games</th>
            </tr>
          </thead>
          <tbody>
            {seasons.map((s) => (
              <tr key={s.season}>
                <td className="py-0.5">{seasonLabel(s.season)}</td>
                <td className="py-0.5 text-right">
                  {s.model.logloss.toFixed(4)} ({s.elo.logloss.toFixed(4)})
                </td>
                <td className="py-0.5 text-right">
                  {s.model.brier.toFixed(4)} ({s.elo.brier.toFixed(4)})
                </td>
                <td className="py-0.5 text-right">
                  {percent(s.model.accuracy, 1)} ({percent(s.elo.accuracy, 1)})
                </td>
                <td className="py-0.5 text-right">
                  {s.model.n.toLocaleString("en-US")}
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </details>
    </div>
  );
}
