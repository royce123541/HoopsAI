import Link from "next/link";

import { GameRow } from "@/components/GameRow";
import { getSlate } from "@/lib/api";
import { longDate, shortDate } from "@/lib/format";

const ISO_DATE = /^\d{4}-\d{2}-\d{2}$/;

export default async function SlatePage({ searchParams }: PageProps<"/">) {
  const { date } = await searchParams;
  const requested =
    typeof date === "string" && ISO_DATE.test(date) ? date : undefined;
  const slate = await getSlate(requested);
  if (!slate) throw new Error("The schedule is unavailable right now.");

  const versions = new Set(
    slate.games.flatMap((g) =>
      g.prediction ? [g.prediction.model_version] : [],
    ),
  );
  const count = slate.games.length;

  return (
    <div className="flex flex-col gap-6">
      <nav
        aria-label="Dates"
        className="flex items-center justify-between gap-4 text-[15px]"
      >
        {slate.prev_date ? (
          <Link
            href={`/?date=${slate.prev_date}`}
            className="text-ink-2 hover:text-ink"
          >
            ‹ {shortDate(slate.prev_date)}
          </Link>
        ) : (
          <span />
        )}
        {slate.next_date ? (
          <Link
            href={`/?date=${slate.next_date}`}
            className="text-ink-2 hover:text-ink"
          >
            {shortDate(slate.next_date)} ›
          </Link>
        ) : (
          <span />
        )}
      </nav>

      <header>
        <h1 className="font-condensed text-5xl font-bold tracking-tight">
          {longDate(slate.date)}
        </h1>
        <p className="mt-1 text-ink-2">
          {count === 0
            ? "No NBA games on this date."
            : `${count} ${count === 1 ? "game" : "games"}${
                versions.size
                  ? `, predictions from model v${[...versions].join(", v")}`
                  : ""
              }`}
        </p>
      </header>

      {count === 0 ? (
        slate.next_date && (
          <p>
            <Link
              href={`/?date=${slate.next_date}`}
              className="font-medium text-ink underline underline-offset-4"
            >
              See the games on {longDate(slate.next_date)}
            </Link>
          </p>
        )
      ) : (
        <>
          <div
            className="flex items-center gap-5 text-sm text-ink-2"
            aria-hidden
          >
            <span className="flex items-center gap-2">
              <span className="h-2.5 w-2.5 rounded-sm bg-away" /> Away
            </span>
            <span className="flex items-center gap-2">
              <span className="h-2.5 w-2.5 rounded-sm bg-home" /> Home
            </span>
            <span className="flex items-center gap-2">
              <span className="h-3 w-px bg-ink-2" /> Even (50%)
            </span>
          </div>
          <ul className="-mx-3 divide-y divide-line border-y border-line">
            {slate.games.map((game) => (
              <GameRow key={game.game_id} game={game} />
            ))}
          </ul>
        </>
      )}
    </div>
  );
}
