import Link from "next/link";
import { notFound } from "next/navigation";

import { EloChart } from "@/components/EloChart";
import { getTeam, getTeams, type TeamGame } from "@/lib/api";
import { percent, shortDate } from "@/lib/format";

function ordinal(n: number): string {
  const s = ["th", "st", "nd", "rd"];
  const v = n % 100;
  return `${n}${s[(v - 20) % 10] ?? s[v] ?? s[0]}`;
}

function GameLine({ game }: { game: TeamGame }) {
  const where = game.is_home ? "vs" : "at";
  const result =
    game.team_score !== null && game.opponent_score !== null
      ? `${game.team_score > game.opponent_score ? "W" : "L"} ${game.team_score}–${game.opponent_score}`
      : null;
  return (
    <li>
      <Link
        href={`/games/${game.game_id}`}
        className="grid grid-cols-[4rem_1fr_auto] gap-3 py-2 text-[15px] hover:bg-surface"
      >
        <span className="tabular text-ink-2">{shortDate(game.game_date)}</span>
        <span className="truncate">
          {where} {game.opponent.full_name}
        </span>
        <span className="tabular text-right">
          {result ??
            (game.win_prob !== null
              ? `${percent(game.win_prob)} to win`
              : "No prediction yet")}
        </span>
      </Link>
    </li>
  );
}

export default async function TeamPage({ params }: PageProps<"/teams/[id]">) {
  const { id } = await params;
  const [team, teams] = await Promise.all([getTeam(id), getTeams()]);
  if (!team) notFound();
  const rank =
    (teams ?? []).findIndex((t) => t.team_id === team.team.team_id) + 1;

  return (
    <div className="flex flex-col gap-10">
      <header>
        <Link href="/teams" className="text-[15px] text-ink-2 hover:text-ink">
          ‹ All teams
        </Link>
        <h1 className="mt-4 font-condensed text-5xl font-bold tracking-tight">
          {team.team.full_name}
        </h1>
        {team.elo !== null && (
          <p className="mt-3 flex items-baseline gap-3">
            <span className="font-condensed text-6xl font-bold leading-none">
              {Math.round(team.elo)}
            </span>
            <span className="text-ink-2">
              Elo rating
              {rank > 0 && `, ${ordinal(rank)} of ${teams?.length ?? 30}`}
            </span>
          </p>
        )}
      </header>

      {team.elo_history.length > 0 && (
        <section aria-labelledby="elo">
          <h2
            id="elo"
            className="font-condensed text-3xl font-bold tracking-tight"
          >
            Elo rating after each game
          </h2>
          <p className="mt-1 mb-4 text-ink-2">
            Last {new Set(team.elo_history.map((p) => p.season)).size} seasons.
          </p>
          <EloChart points={team.elo_history} />
          <details className="mt-3 text-sm">
            <summary className="cursor-pointer text-ink-2">
              Show as a table
            </summary>
            <table className="mt-2 w-full max-w-sm">
              <thead>
                <tr className="text-left text-ink-2">
                  <th className="py-1 font-medium">Date</th>
                  <th className="py-1 text-right font-medium">Elo</th>
                </tr>
              </thead>
              <tbody className="tabular">
                {team.elo_history.map((p) => (
                  <tr key={p.game_date}>
                    <td className="py-0.5">{shortDate(p.game_date, true)}</td>
                    <td className="py-0.5 text-right">{Math.round(p.elo)}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </details>
        </section>
      )}

      <div className="grid gap-10 sm:grid-cols-2">
        <section aria-labelledby="upcoming">
          <h2
            id="upcoming"
            className="mb-2 font-condensed text-3xl font-bold tracking-tight"
          >
            Next games
          </h2>
          {team.upcoming.length ? (
            <ul className="divide-y divide-line border-y border-line">
              {team.upcoming.map((g) => (
                <GameLine key={g.game_id} game={g} />
              ))}
            </ul>
          ) : (
            <p className="text-ink-2">No games scheduled.</p>
          )}
        </section>
        <section aria-labelledby="recent">
          <h2
            id="recent"
            className="mb-2 font-condensed text-3xl font-bold tracking-tight"
          >
            Recent results
          </h2>
          <ul className="divide-y divide-line border-y border-line">
            {team.recent.map((g) => (
              <GameLine key={g.game_id} game={g} />
            ))}
          </ul>
        </section>
      </div>
    </div>
  );
}
