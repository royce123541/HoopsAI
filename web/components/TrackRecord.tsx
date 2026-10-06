import type { MonitoringResult } from "@/lib/api";
import { percent } from "@/lib/format";

const MODEL_LABEL = {
  pregame: "Before tip-off",
  ingame: "During games",
} as const;
const WINDOW_LABEL = {
  season: "This season",
  last_30_days: "Last 30 days",
} as const;

// Status always pairs a symbol with a word; colour is never the only signal.
const STATUS = {
  ok: { symbol: "✓", label: "On track", className: "text-ink" },
  warning: {
    symbol: "!",
    label: "Worse than expected",
    className: "font-semibold text-ink",
  },
  insufficient: {
    symbol: "–",
    label: "Not enough games yet",
    className: "text-ink-2",
  },
} as const;

/** How the models are doing on real games this season, against their held-out evaluation. */
export function TrackRecord({ results }: { results: MonitoringResult[] }) {
  if (results.length === 0) {
    return (
      <p className="max-w-prose text-ink-2">
        No checks have run yet. They run each night after the day&apos;s games
        are loaded.
      </p>
    );
  }
  const checked = results[0].run_at;
  return (
    <div>
      {/* Five columns: scroll sideways on narrow phones rather than squeeze. */}
      <div className="overflow-x-auto">
        <table className="w-full min-w-[34rem] text-[15px]">
          <thead>
            <tr className="border-b border-line text-left text-sm text-ink-2">
              <th scope="col" className="py-2 font-medium">
                Predictions
              </th>
              <th scope="col" className="py-2 font-medium">
                Games
              </th>
              <th scope="col" className="py-2 text-right font-medium">
                Log loss (expected)
              </th>
              <th scope="col" className="py-2 text-right font-medium">
                Winner right
              </th>
              <th scope="col" className="py-2 pl-6 font-medium">
                Status
              </th>
            </tr>
          </thead>
          <tbody className="divide-y divide-line">
            {results.map((r) => {
              const s = STATUS[r.status];
              return (
                <tr key={`${r.model}-${r.window}`} title={r.message}>
                  <th scope="row" className="py-2 text-left font-normal">
                    {MODEL_LABEL[r.model]}
                    <span className="block text-sm text-ink-2">
                      {WINDOW_LABEL[r.window]}
                    </span>
                  </th>
                  <td className="tabular py-2">
                    {r.games.toLocaleString("en-US")}
                  </td>
                  <td className="tabular py-2 text-right">
                    {r.metrics ? r.metrics.logloss.toFixed(3) : "—"}
                    {r.expected_logloss !== null && (
                      <span className="text-ink-2">
                        {" "}
                        ({r.expected_logloss.toFixed(3)})
                      </span>
                    )}
                  </td>
                  <td className="tabular py-2 text-right">
                    {r.metrics ? percent(r.metrics.accuracy, 1) : "—"}
                  </td>
                  <td className={`py-2 pl-6 ${s.className}`}>
                    <span
                      aria-hidden
                      className="mr-1.5 inline-block w-3 text-center"
                    >
                      {s.symbol}
                    </span>
                    {s.label}
                  </td>
                </tr>
              );
            })}
          </tbody>
        </table>
      </div>
      <p className="mt-2 text-sm text-muted">
        Only predictions made before tip-off count, and only live in-game
        updates (not replays). Last checked{" "}
        {new Intl.DateTimeFormat("en-US", {
          dateStyle: "medium",
          timeStyle: "short",
          timeZone: "America/New_York",
        }).format(new Date(checked))}{" "}
        ET.
      </p>
    </div>
  );
}
