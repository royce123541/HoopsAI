import type { ComparisonRow, TeamRef } from "@/lib/api";
import { featureValue } from "@/lib/format";

function better(row: ComparisonRow): "home" | "away" | null {
  if (row.home === null || row.away === null || row.home === row.away)
    return null;
  const homeHigher = row.home > row.away;
  return homeHigher === row.higher_is_better ? "home" : "away";
}

/** Away and home side by side, in the same order as the matchup; the edge is set bold. */
export function ComparisonTable({
  rows,
  home,
  away,
}: {
  rows: ComparisonRow[];
  home: TeamRef;
  away: TeamRef;
}) {
  const missing = rows.some((r) => r.home === null || r.away === null);
  return (
    <div>
      <table className="w-full text-[15px]">
        <thead>
          <tr className="border-b border-line text-left text-sm text-ink-2">
            <th scope="col" className="py-2 font-medium">
              Before tip-off
            </th>
            <th scope="col" className="w-24 py-2 text-right font-medium">
              {away.abbreviation}
            </th>
            <th scope="col" className="w-24 py-2 text-right font-medium">
              {home.abbreviation}
            </th>
          </tr>
        </thead>
        <tbody className="divide-y divide-line">
          {rows.map((row) => {
            const edge = better(row);
            const key = `away_${row.key}`;
            return (
              <tr key={row.key}>
                <th
                  scope="row"
                  className="py-2 text-left font-normal text-ink-2"
                >
                  {row.label}
                </th>
                <td
                  className={`tabular py-2 text-right ${edge === "away" ? "font-semibold text-ink" : ""}`}
                >
                  {featureValue(key, row.away)}
                </td>
                <td
                  className={`tabular py-2 text-right ${edge === "home" ? "font-semibold text-ink" : ""}`}
                >
                  {featureValue(key, row.home)}
                </td>
              </tr>
            );
          })}
        </tbody>
      </table>
      {missing && (
        <p className="mt-2 text-sm text-muted">
          — means the team hasn&apos;t played enough games this season yet.
        </p>
      )}
    </div>
  );
}
