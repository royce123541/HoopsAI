import Link from "next/link";

import type { GameSummary } from "@/lib/api";
import { gameClock, splitPercents, tipTime } from "@/lib/format";
import { SplitBar } from "./SplitBar";

function statusText(game: GameSummary): string {
  if (game.status === "final") return "Final";
  if (game.status === "live")
    return game.live
      ? `Live, ${gameClock(game.live.period, game.live.clock_seconds)}`
      : "Live";
  return tipTime(game.tip_time_utc);
}

/** One sentence under the bar: the matchup, or for a finished game, how it went. */
function caption(game: GameSummary): string {
  const matchup = `${game.away.nickname} at ${game.home.nickname}`;
  const kind =
    game.season_type === "playoffs"
      ? ", playoffs"
      : game.season_type === "playin"
        ? ", play-in"
        : "";
  if (
    game.status !== "final" ||
    game.home_score === null ||
    game.away_score === null
  ) {
    return `${matchup}${game.is_neutral ? ", neutral site" : ""}${kind}`;
  }
  const homeWon = game.home_score > game.away_score;
  const winner = homeWon ? game.home : game.away;
  const score = homeWon
    ? `${game.home_score}–${game.away_score}`
    : `${game.away_score}–${game.home_score}`;
  if (!game.prediction) return `${winner.nickname} won ${score}`;
  const pct = splitPercents(game.prediction.home_win_prob);
  const winnerPct = homeWon ? pct.home : pct.away;
  return `${winner.nickname} won ${score}. The model gave them ${winnerPct}%.`;
}

export function GameRow({ game }: { game: GameSummary }) {
  // While a game is live the bar follows the in-game estimate, not the pre-game one.
  const prob = game.live?.home_win_prob ?? game.prediction?.home_win_prob;
  return (
    <li>
      <Link
        href={`/games/${game.game_id}`}
        className="grid gap-x-6 gap-y-1 rounded-md px-3 py-4 hover:bg-surface sm:grid-cols-[7rem_1fr] sm:items-center"
      >
        <span className="text-sm font-medium text-ink-2 sm:pt-1">
          {statusText(game)}
          {game.status !== "scheduled" && game.home_score !== null && (
            <span className="tabular ml-2 text-ink">
              {game.away_score}–{game.home_score}
            </span>
          )}
        </span>
        <span className="block min-w-0">
          {prob !== undefined ? (
            <SplitBar home={game.home} away={game.away} homeProb={prob} />
          ) : (
            <span className="flex items-baseline justify-between py-2 font-condensed text-lg font-semibold text-ink-2">
              <span>{game.away.abbreviation}</span>
              <span className="font-sans text-sm font-normal text-muted">
                No prediction yet
              </span>
              <span>{game.home.abbreviation}</span>
            </span>
          )}
          <span className="block text-sm text-ink-2">{caption(game)}</span>
        </span>
      </Link>
    </li>
  );
}
