// NBA schedules are published in US Eastern time; tip times are shown in ET everywhere.
const ET = "America/New_York";

/** Whole-number percentages for both teams that always add up to 100. */
export function splitPercents(homeProb: number): {
  home: number;
  away: number;
} {
  const home = Math.round(homeProb * 100);
  return { home, away: 100 - home };
}

export function percent(p: number, digits = 0): string {
  return `${(p * 100).toFixed(digits)}%`;
}

export function tipTime(iso: string | null): string {
  if (!iso) return "Time TBD";
  const time = new Intl.DateTimeFormat("en-US", {
    hour: "numeric",
    minute: "2-digit",
    timeZone: ET,
  }).format(new Date(iso));
  return `${time} ET`;
}

// Calendar dates ("2026-10-20") are dates, not instants: format them at noon UTC so no
// timezone can move them to a neighbouring day.
function calendarDate(isoDate: string): Date {
  return new Date(`${isoDate}T12:00:00Z`);
}

export function longDate(isoDate: string): string {
  return new Intl.DateTimeFormat("en-US", {
    weekday: "long",
    month: "long",
    day: "numeric",
    timeZone: "UTC",
  }).format(calendarDate(isoDate));
}

export function shortDate(isoDate: string, withYear = false): string {
  return new Intl.DateTimeFormat("en-US", {
    month: "short",
    day: "numeric",
    ...(withYear ? { year: "numeric" } : {}),
    timeZone: "UTC",
  }).format(calendarDate(isoDate));
}

export function seasonLabel(season: number): string {
  return `${season}-${String((season + 1) % 100).padStart(2, "0")}`;
}

/** Display a model feature's value in its natural unit. */
export function featureValue(
  feature: string,
  value: number | null | undefined,
): string {
  if (value === null || value === undefined) return "—";
  const base = feature.replace(/^(home_|away_|diff_)/, "");
  const signed = feature.startsWith("diff_");
  const sign = signed && value > 0 ? "+" : "";
  if (/(win_pct|efg|tov_pct|orb_pct|drb_pct|ftr)$/.test(base)) {
    return `${sign}${(value * 100).toFixed(1)}%`;
  }
  if (base === "elo_pre" || base === "travel_km")
    return `${sign}${Math.round(value).toLocaleString("en-US")}`;
  if (
    base === "rest_days" ||
    base === "games_played" ||
    base === "streak" ||
    base === "games_last7"
  ) {
    return `${sign}${Math.round(value)}`;
  }
  if (base === "b2b" || base === "is_neutral" || base === "is_playoffs")
    return value ? "Yes" : "No";
  return `${sign}${value.toFixed(1)}`;
}
