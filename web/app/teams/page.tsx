import Link from "next/link";

import { getTeams } from "@/lib/api";

const LEAGUE_AVERAGE = 1505;

/** Power ranking by Elo: a dot per team on one shared scale, league average marked. */
export default async function TeamsPage() {
  const teams = (await getTeams()) ?? [];
  const rated = teams.filter((t) => t.elo !== null);
  const elos = rated.map((t) => t.elo as number);
  const lo = Math.floor((Math.min(...elos, LEAGUE_AVERAGE) - 20) / 50) * 50;
  const hi = Math.ceil((Math.max(...elos, LEAGUE_AVERAGE) + 20) / 50) * 50;
  const x = (elo: number) => `${((elo - lo) / (hi - lo)) * 100}%`;

  return (
    <div className="flex flex-col gap-6">
      <header>
        <h1 className="font-condensed text-5xl font-bold tracking-tight">
          Teams by Elo rating
        </h1>
        <p className="mt-1 max-w-prose text-ink-2">
          Elo rates each team from its results, weighted by margin and opponent
          strength. It is the single strongest input to the win-probability
          model.
        </p>
      </header>

      <ol className="divide-y divide-line border-y border-line">
        {rated.map((team, i) => (
          <li key={team.team_id}>
            <Link
              href={`/teams/${team.team_id}`}
              className="grid grid-cols-[2rem_minmax(0,11rem)_3.5rem_1fr] items-center gap-3 py-2 hover:bg-surface"
            >
              <span className="tabular text-right text-sm text-muted">
                {i + 1}
              </span>
              <span className="truncate text-[15px] text-ink">
                {team.full_name}
              </span>
              <span className="tabular text-right text-[15px] font-semibold">
                {Math.round(team.elo as number)}
              </span>
              <span aria-hidden className="relative block h-4">
                <span
                  className="absolute inset-y-0 w-px bg-axis"
                  style={{ left: x(LEAGUE_AVERAGE) }}
                />
                <span
                  className="absolute top-1/2 h-2.5 w-2.5 -translate-x-1/2 -translate-y-1/2 rounded-full bg-home ring-2 ring-[var(--page)]"
                  style={{ left: x(team.elo as number) }}
                />
              </span>
            </Link>
          </li>
        ))}
      </ol>
      <p className="text-sm text-muted">
        The vertical line marks the league average of {LEAGUE_AVERAGE}.
      </p>
    </div>
  );
}
