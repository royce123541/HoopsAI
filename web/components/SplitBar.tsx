import type { TeamRef } from "@/lib/api";
import { splitPercents } from "@/lib/format";

type Props = {
  home: TeamRef;
  away: TeamRef;
  homeProb: number;
  size?: "row" | "hero";
};

/**
 * Win probability as a court split: away (orange) left, home (blue) right, meeting at the
 * share each team is expected to win. The mid-court line marks 50%, so the eye reads the
 * favourite by which side of it the meeting point falls. Percentages sit at the ends in
 * ink (never in the series colour); the favourite's is set heavier.
 */
export function SplitBar({ home, away, homeProb, size = "row" }: Props) {
  const pct = splitPercents(homeProb);
  const hero = size === "hero";
  const homeFavored = pct.home >= pct.away;
  const numberClass = hero
    ? "font-condensed text-6xl leading-none sm:text-7xl"
    : "font-condensed text-3xl leading-none";
  const abbrClass = hero ? "font-condensed text-2xl" : "font-condensed text-lg";
  // Wide enough for "100%" at each size, so a number never runs into the bar.
  const sideClass = hero ? "w-28 sm:w-40" : "w-[4.5rem] sm:w-24";
  const label = `${away.abbreviation} ${pct.away}% to win, ${home.abbreviation} ${pct.home}% to win`;

  return (
    <div className="flex items-center gap-3 sm:gap-4">
      <div className={`${sideClass} shrink-0 text-left`}>
        <div className={`${abbrClass} font-semibold text-ink-2`}>
          {away.abbreviation}
        </div>
        <div
          className={`${numberClass} ${homeFavored ? "font-medium text-ink-2" : "font-bold text-ink"}`}
        >
          {pct.away}%
        </div>
      </div>

      <div
        role="img"
        aria-label={label}
        className="relative min-w-0 flex-1 py-3"
      >
        <div className={`flex gap-[2px] ${hero ? "h-5" : "h-3"}`}>
          <span
            className="grow-from-right rounded-l bg-away"
            style={{ flexGrow: pct.away, flexBasis: 0 }}
          />
          <span
            className="grow-from-left rounded-r bg-home"
            style={{ flexGrow: pct.home, flexBasis: 0 }}
          />
        </div>
        {/* Mid-court line: 50% */}
        <span
          aria-hidden
          className="absolute inset-y-0 left-1/2 w-px -translate-x-1/2 bg-ink-2"
        />
      </div>

      <div className={`${sideClass} shrink-0 text-right`}>
        <div className={`${abbrClass} font-semibold text-ink-2`}>
          {home.abbreviation}
        </div>
        <div
          className={`${numberClass} ${homeFavored ? "font-bold text-ink" : "font-medium text-ink-2"}`}
        >
          {pct.home}%
        </div>
      </div>
    </div>
  );
}
