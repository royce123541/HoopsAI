import type { Factor, TeamRef } from "@/lib/api";
import { featureValue } from "@/lib/format";

/**
 * Diverging bars from a centre baseline: a factor pushing toward the away team extends left
 * in orange, toward the home team right in blue. Length = size of its push (TreeSHAP
 * contribution), scaled to the largest factor shown.
 */
export function FactorBars({
  factors,
  home,
  away,
}: {
  factors: Factor[];
  home: TeamRef;
  away: TeamRef;
}) {
  if (factors.length === 0) {
    return (
      <p className="text-ink-2">
        No factor breakdown is available for this prediction.
      </p>
    );
  }
  const max = Math.max(...factors.map((f) => Math.abs(f.contribution)));
  return (
    <div>
      <div className="mb-3 flex gap-5 text-sm text-ink-2">
        <span className="flex items-center gap-2">
          <span aria-hidden className="h-2.5 w-2.5 rounded-sm bg-away" /> Favors{" "}
          {away.abbreviation}
        </span>
        <span className="flex items-center gap-2">
          <span aria-hidden className="h-2.5 w-2.5 rounded-sm bg-home" /> Favors{" "}
          {home.abbreviation}
        </span>
      </div>
      <ul className="divide-y divide-line border-y border-line">
        {factors.map((f) => {
          const width = `${(Math.abs(f.contribution) / max) * 50}%`;
          const favored = f.favors === "home" ? home : away;
          return (
            <li
              key={f.feature}
              className="grid grid-cols-1 gap-x-4 gap-y-1 py-3 sm:grid-cols-[minmax(0,1fr)_minmax(0,1fr)] sm:items-center"
            >
              <span className="text-[15px]">
                <span className="text-ink">{f.label}</span>
                <span className="tabular ml-2 text-ink-2">
                  {featureValue(f.feature, f.value)}
                </span>
              </span>
              <span
                className="relative block h-3"
                role="img"
                aria-label={`Favors ${favored.abbreviation}`}
                title={`Favors ${favored.abbreviation}`}
              >
                <span
                  aria-hidden
                  className="absolute inset-y-[-4px] left-1/2 w-px bg-axis"
                />
                <span
                  className={`absolute inset-y-0 ${
                    f.favors === "home"
                      ? "left-1/2 ml-px rounded-r bg-home"
                      : "right-1/2 mr-px rounded-l bg-away"
                  }`}
                  style={{ width }}
                />
              </span>
            </li>
          );
        })}
      </ul>
    </div>
  );
}
