import Link from "next/link";
import { notFound } from "next/navigation";

import { ComparisonTable } from "@/components/ComparisonTable";
import { FactorBars } from "@/components/FactorBars";
import { SplitBar } from "@/components/SplitBar";
import { getGame } from "@/lib/api";
import { longDate, shortDate, splitPercents, tipTime } from "@/lib/format";

export default async function GamePage({ params }: PageProps<"/games/[id]">) {
  const { id } = await params;
  const game = await getGame(id);
  if (!game) notFound();

  const { home, away, prediction } = game;
  const pct = prediction ? splitPercents(prediction.home_win_prob) : null;
  const favorite = pct && (pct.home >= pct.away ? home : away);
  const final =
    game.status === "final" &&
    game.home_score !== null &&
    game.away_score !== null;

  return (
    <article className="flex flex-col gap-10">
      <div>
        <Link
          href={`/?date=${game.game_date}`}
          className="text-[15px] text-ink-2 hover:text-ink"
        >
          ‹ All games on {shortDate(game.game_date)}
        </Link>
        <h1 className="mt-4 font-condensed text-5xl font-bold tracking-tight">
          {away.nickname} at {home.nickname}
        </h1>
        <p className="mt-1 text-ink-2">
          {longDate(game.game_date)}, {tipTime(game.tip_time_utc)}
          {game.is_neutral && ", neutral site"}
          {game.season_type === "playoffs" && ", playoffs"}
          {game.season_type === "playin" && ", play-in tournament"}
        </p>
      </div>

      <section aria-labelledby="chance">
        <h2 id="chance" className="sr-only">
          Chance to win
        </h2>
        {prediction && pct && favorite ? (
          <>
            <SplitBar
              home={home}
              away={away}
              homeProb={prediction.home_win_prob}
              size="hero"
            />
            <p className="mt-3 text-ink-2">
              {pct.home === pct.away
                ? "The model sees this game as a coin flip."
                : `The model gives the ${favorite.nickname} a ${Math.max(pct.home, pct.away)}% chance to win.`}
              {final &&
                ` Final score: ${away.abbreviation} ${game.away_score}, ${home.abbreviation} ${game.home_score}.`}
            </p>
            {!prediction.made_before_tip && (
              <p className="mt-1 text-sm text-muted">
                This prediction was made after tip-off from pre-game data, so it
                is not a forecast the model made in advance.
              </p>
            )}
          </>
        ) : (
          <p className="text-ink-2">
            No prediction yet. Predictions are made for games in the coming
            week.
          </p>
        )}
      </section>

      {prediction && (
        <section aria-labelledby="why">
          <h2
            id="why"
            className="font-condensed text-3xl font-bold tracking-tight"
          >
            What drives the prediction
          </h2>
          <p className="mt-1 mb-4 max-w-prose text-ink-2">
            The factors that moved the model&apos;s estimate the most, and which
            team each one helps.
          </p>
          <FactorBars factors={game.factors} home={home} away={away} />
        </section>
      )}

      <section aria-labelledby="compare">
        <h2
          id="compare"
          className="mb-4 font-condensed text-3xl font-bold tracking-tight"
        >
          How they compare
        </h2>
        <ComparisonTable rows={game.comparison} home={home} away={away} />
      </section>

      <p className="text-[15px]">
        <Link
          href={`/teams/${away.team_id}`}
          className="text-ink underline underline-offset-4"
        >
          {away.full_name}
        </Link>
        <span className="text-ink-2"> and </span>
        <Link
          href={`/teams/${home.team_id}`}
          className="text-ink underline underline-offset-4"
        >
          {home.full_name}
        </Link>
        <span className="text-ink-2"> team pages</span>
      </p>
    </article>
  );
}
