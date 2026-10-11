"""Pure parsers: ESPN site-API JSON -> HoopsAI rows. No network, no database.

ESPN's public JSON (site.api.espn.com) answers from cloud servers, where stats.nba.com times
out, so daily updates and live tracking can run on GitHub Actions. Teams map to NBA team ids
(the database's keys); plays are translated into the PlayByPlayV3 event shape so the in-game
model's possession rules (hoopsai.live.state) apply unchanged."""

import re
from dataclasses import dataclass
from datetime import date, datetime
from typing import Any

from hoopsai.dates import EASTERN
from hoopsai.sources.base import GameStatus, PbpEventRow, SeasonType

# ESPN team id -> NBA team id (franchise ids are stable, see NbaStatsSource.list_teams).
ESPN_TO_NBA_TEAM = {
    "1": 1610612737,  # ATL
    "2": 1610612738,  # BOS
    "17": 1610612751,  # BKN
    "30": 1610612766,  # CHA
    "4": 1610612741,  # CHI
    "5": 1610612739,  # CLE
    "6": 1610612742,  # DAL
    "7": 1610612743,  # DEN
    "8": 1610612765,  # DET
    "9": 1610612744,  # GSW
    "10": 1610612745,  # HOU
    "11": 1610612754,  # IND
    "12": 1610612746,  # LAC
    "13": 1610612747,  # LAL
    "29": 1610612763,  # MEM
    "14": 1610612748,  # MIA
    "15": 1610612749,  # MIL
    "16": 1610612750,  # MIN
    "3": 1610612740,  # NOP
    "18": 1610612752,  # NYK
    "25": 1610612760,  # OKC
    "19": 1610612753,  # ORL
    "20": 1610612755,  # PHI
    "21": 1610612756,  # PHX
    "22": 1610612757,  # POR
    "23": 1610612758,  # SAC
    "24": 1610612759,  # SAS
    "28": 1610612761,  # TOR
    "26": 1610612762,  # UTA
    "27": 1610612764,  # WAS
}

# ESPN season.type -> ours; preseason (1), All-Star and others are skipped.
SEASON_TYPES = {2: SeasonType.REGULAR, 3: SeasonType.PLAYOFFS, 5: SeasonType.PLAYIN}
STATUS = {"pre": GameStatus.SCHEDULED, "in": GameStatus.LIVE, "post": GameStatus.FINAL}
# A postponed game keeps its original slot as a 0-0 "post" entry; its makeup is a new event.
NOT_PLAYED = {"STATUS_POSTPONED", "STATUS_CANCELED"}
# The NBA Cup final does not count in the NBA's records (stats.nba.com omits it), so skip it.
EXHIBITION_TYPES = {"CC"}

_FT = re.compile(r"Free Throw - (.*)")


@dataclass(frozen=True)
class EspnGame:
    espn_id: str
    season: int  # start year: ESPN's 2026 season is our 2025 (2025-26)
    season_type: SeasonType
    game_date: date  # US Eastern calendar date
    tip_time_utc: datetime
    home_team_id: int
    away_team_id: int
    status: GameStatus
    home_score: int | None
    away_score: int | None
    is_neutral: bool
    period: int
    clock: str


def parse_scoreboard(payload: dict[str, Any]) -> list[EspnGame]:
    games = []
    for ev in payload.get("events", []):
        season_type = SEASON_TYPES.get(int(ev["season"]["type"]))
        comp = ev["competitions"][0]
        teams = {c["homeAway"]: c for c in comp["competitors"]}
        home_id = ESPN_TO_NBA_TEAM.get(teams.get("home", {}).get("team", {}).get("id", ""))
        away_id = ESPN_TO_NBA_TEAM.get(teams.get("away", {}).get("team", {}).get("id", ""))
        if season_type is None or home_id is None or away_id is None:
            continue  # preseason, All-Star, or an unresolved (TBD) playoff slot
        if (
            ev["status"]["type"]["name"] in NOT_PLAYED
            or comp.get("type", {}).get("abbreviation") in EXHIBITION_TYPES
        ):
            continue
        status = STATUS.get(ev["status"]["type"]["state"], GameStatus.SCHEDULED)
        tip = datetime.fromisoformat(ev["date"].replace("Z", "+00:00"))
        scored = status is not GameStatus.SCHEDULED
        games.append(
            EspnGame(
                espn_id=str(ev["id"]),
                season=int(ev["season"]["year"]) - 1,
                season_type=season_type,
                game_date=tip.astimezone(EASTERN).date(),
                tip_time_utc=tip,
                home_team_id=home_id,
                away_team_id=away_id,
                status=status,
                home_score=int(teams["home"]["score"]) if scored else None,
                away_score=int(teams["away"]["score"]) if scored else None,
                is_neutral=bool(comp.get("neutralSite")),
                period=int(ev["status"].get("period") or 0),
                clock=str(ev["status"].get("displayClock") or ""),
            )
        )
    return games


def _pair(stats: dict[str, str], name: str) -> tuple[int, int]:
    made, attempted = stats[name].split("-")
    return int(made), int(attempted)


def parse_team_box(summary: dict[str, Any], game_id: str) -> list[dict[str, Any]]:
    """Team box scores in core.team_game_stats shape. `totalTurnovers` includes team
    turnovers, matching the NBA's TOV; minutes come from the number of periods played."""
    comp = summary["header"]["competitions"][0]
    home_espn = next(c["team"]["id"] for c in comp["competitors"] if c["homeAway"] == "home")
    periods = max(4, max((p["period"]["number"] for p in summary.get("plays", [])), default=4))
    minutes = 240 + 25 * (periods - 4)
    points = {c["team"]["id"]: int(c["score"]) for c in comp["competitors"]}
    rows = []
    for team in summary["boxscore"]["teams"]:
        espn_id = team["team"]["id"]
        s = {x["name"]: x["displayValue"] for x in team["statistics"]}
        fgm, fga = _pair(s, "fieldGoalsMade-fieldGoalsAttempted")
        fg3m, fg3a = _pair(s, "threePointFieldGoalsMade-threePointFieldGoalsAttempted")
        ftm, fta = _pair(s, "freeThrowsMade-freeThrowsAttempted")
        rows.append(
            {
                "game_id": game_id,
                "team_id": ESPN_TO_NBA_TEAM[espn_id],
                "is_home": espn_id == home_espn,
                "minutes": minutes,
                "pts": points[espn_id],
                "fgm": fgm,
                "fga": fga,
                "fg3m": fg3m,
                "fg3a": fg3a,
                "ftm": ftm,
                "fta": fta,
                "oreb": int(s["offensiveRebounds"]),
                "dreb": int(s["defensiveRebounds"]),
                "reb": int(s["totalRebounds"]),
                "ast": int(s["assists"]),
                "stl": int(s["steals"]),
                "blk": int(s["blocks"]),
                "tov": int(s.get("totalTurnovers") or s["turnovers"]),
                "pf": int(s["fouls"]),
            }
        )
    home, away = rows if rows[0]["is_home"] else rows[::-1]
    home["plus_minus"], away["plus_minus"] = home["pts"] - away["pts"], away["pts"] - home["pts"]
    return rows


def parse_clock(display: str) -> float:
    """ESPN clocks: "11:35" (minutes:seconds) or "45.3" (seconds, under a minute)."""
    if ":" in display:
        minutes, seconds = display.split(":")
        return int(minutes) * 60 + float(seconds)
    try:
        return float(display)
    except ValueError:
        return 0.0


def _action(play: dict[str, Any]) -> tuple[str | None, str | None, str | None]:
    """(action_type, sub_type, description) in PlayByPlayV3 vocabulary."""
    kind = play["type"]["text"].replace("\n", " ")
    made = bool(play.get("scoringPlay"))
    if kind.startswith("Free Throw"):
        m = _FT.match(kind)
        detail = (m.group(1) if m else "").replace("- ", "")
        return "Free Throw", f"Free Throw {detail}".strip(), None if made else "MISS"
    if "Rebound" in kind:
        return "Rebound", None, None
    if "Turnover" in kind:
        return "Turnover", None, None
    if kind == "Jumpball":
        return "Jump Ball", None, None
    if kind in ("End Period", "End Game", "Start Period"):
        return "period", None, None
    if kind == "Substitution":
        return "Substitution", None, None
    if "Timeout" in kind:
        return "Timeout", None, None
    if "Review" in kind or "Challenge" in kind:
        return "Instant Replay", None, None
    if play.get("shootingPlay"):
        return ("Made Shot" if made else "Missed Shot"), kind, None
    return "Foul" if "Foul" in kind else None, kind, None


def parse_plays(summary: dict[str, Any], game_id: str) -> list[PbpEventRow]:
    """ESPN plays -> PbpEventRow in feed order (action_id = position, 1..N)."""
    comp = summary["header"]["competitions"][0]
    home_espn = next(c["team"]["id"] for c in comp["competitors"] if c["homeAway"] == "home")
    rows: list[PbpEventRow] = []
    for i, p in enumerate(summary.get("plays", []), start=1):
        action_type, sub_type, description = _action(p)
        team = (p.get("team") or {}).get("id")
        rows.append(
            PbpEventRow(
                game_id=game_id,
                action_id=i,
                action_number=int(p.get("sequenceNumber") or i),
                period=int(p["period"]["number"]),
                clock_seconds=parse_clock(p["clock"]["displayValue"]),
                team_id=ESPN_TO_NBA_TEAM.get(team) if team else None,
                location=None if not team else ("h" if team == home_espn else "v"),
                person_id=None,
                action_type=action_type,
                sub_type=sub_type,
                description=description or p.get("text"),
                is_field_goal=bool(p.get("shootingPlay")) and action_type != "Free Throw",
                shot_value=int(p["scoreValue"]) if p.get("scoringPlay") else None,
                shot_result=None,
                score_home=int(p["homeScore"]),
                score_away=int(p["awayScore"]),
            )
        )
    return rows
