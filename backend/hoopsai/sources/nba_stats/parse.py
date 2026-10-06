"""Pure parsers: stats.nba.com JSON payloads -> core-table rows. No network, no database."""

import logging
import re
from collections import defaultdict
from collections.abc import Mapping
from datetime import date, datetime
from typing import Any

from hoopsai.sources.base import (
    GameLogParse,
    GameRow,
    GameStatus,
    PbpEventRow,
    ScheduledGameRow,
    SeasonType,
    TeamGameStatsRow,
    parse_season,
)

log = logging.getLogger(__name__)

# The 3-digit game-id prefix encodes the game type. Preseason (001), All-Star (003) and
# the NBA Cup final (006, no box score in the season game logs) are deliberately excluded.
SEASON_TYPE_BY_PREFIX = {
    "002": SeasonType.REGULAR,
    "004": SeasonType.PLAYOFFS,
    "005": SeasonType.PLAYIN,
}

SCHEDULE_STATUS = {1: GameStatus.SCHEDULED, 2: GameStatus.LIVE, 3: GameStatus.FINAL}

_CLOCK_RE = re.compile(r"PT(\d+)M(\d+(?:\.\d+)?)S")


def season_type_of(game_id: str) -> SeasonType | None:
    return SEASON_TYPE_BY_PREFIX.get(game_id[:3])


def _result_set(payload: dict[str, Any]) -> list[dict[str, Any]]:
    result = payload["resultSets"][0]
    headers: list[str] = result["headers"]
    return [dict(zip(headers, row, strict=True)) for row in result["rowSet"]]


def parse_team_game_logs(
    payload: dict[str, Any], home_team_ids: Mapping[str, int] | None = None
) -> GameLogParse:
    """LeagueGameLog (PlayerOrTeam=T) has one row per team per game; pair them into games.
    MATCHUP is "BOS vs. NYK" for the home team and "NYK @ BOS" for the away team.

    Neutral-site games (international games, NBA Cup knockouts) list both teams with "@";
    `home_team_ids` (game_id -> designated home team, from the schedule) resolves them."""
    home_team_ids = home_team_ids or {}
    by_game: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in _result_set(payload):
        if season_type_of(row["GAME_ID"]) is not None:
            by_game[row["GAME_ID"]].append(row)

    games: list[GameRow] = []
    team_stats: list[TeamGameStatsRow] = []
    for game_id, rows in by_game.items():
        home = [r for r in rows if " vs. " in r["MATCHUP"]]
        away = [r for r in rows if " @ " in r["MATCHUP"]]
        if len(rows) == 2 and (len(home), len(away)) != (1, 1) and game_id in home_team_ids:
            home = [r for r in rows if r["TEAM_ID"] == home_team_ids[game_id]]
            away = [r for r in rows if r["TEAM_ID"] != home_team_ids[game_id]]
        if len(rows) != 2 or len(home) != 1 or len(away) != 1:
            log.warning("skipping game %s: expected one home and one away row", game_id)
            continue
        # Cancelled games (e.g. BOS-IND, 2013-04-16) appear with no result and 0 minutes.
        if any(r["WL"] is None or not r["MIN"] for r in rows):
            log.warning("skipping game %s: not played", game_id)
            continue
        try:
            stats = [_team_stats(game_id, home[0], True), _team_stats(game_id, away[0], False)]
        except (TypeError, ValueError):
            log.warning("skipping game %s: incomplete box score", game_id)
            continue
        season_type = season_type_of(game_id)
        assert season_type is not None
        games.append(
            GameRow(
                game_id=game_id,
                season=int(home[0]["SEASON_ID"][1:]),
                season_type=season_type,
                game_date=date.fromisoformat(home[0]["GAME_DATE"][:10]),
                home_team_id=home[0]["TEAM_ID"],
                away_team_id=away[0]["TEAM_ID"],
                status=GameStatus.FINAL,
                home_score=stats[0]["pts"],
                away_score=stats[1]["pts"],
            )
        )
        team_stats.extend(stats)
    return GameLogParse(games=games, team_stats=team_stats)


def _team_stats(game_id: str, row: dict[str, Any], is_home: bool) -> TeamGameStatsRow:
    """Raises TypeError/ValueError if any box-score value is missing or non-numeric."""
    return TeamGameStatsRow(
        game_id=game_id,
        team_id=row["TEAM_ID"],
        is_home=is_home,
        minutes=int(row["MIN"]),
        pts=int(row["PTS"]),
        fgm=int(row["FGM"]),
        fga=int(row["FGA"]),
        fg3m=int(row["FG3M"]),
        fg3a=int(row["FG3A"]),
        ftm=int(row["FTM"]),
        fta=int(row["FTA"]),
        oreb=int(row["OREB"]),
        dreb=int(row["DREB"]),
        reb=int(row["REB"]),
        ast=int(row["AST"]),
        stl=int(row["STL"]),
        blk=int(row["BLK"]),
        tov=int(row["TOV"]),
        pf=int(row["PF"]),
        plus_minus=int(row["PLUS_MINUS"]),
    )


def parse_schedule(payload: dict[str, Any]) -> list[ScheduledGameRow]:
    schedule = payload["leagueSchedule"]
    season = parse_season(schedule["seasonYear"])
    rows: list[ScheduledGameRow] = []
    for game_date in schedule["gameDates"]:
        for g in game_date["games"]:
            season_type = season_type_of(g["gameId"])
            home_id, away_id = g["homeTeam"]["teamId"], g["awayTeam"]["teamId"]
            # Unresolved playoff/play-in slots ("if necessary", TBD opponents) have team id 0.
            if season_type is None or not home_id or not away_id:
                continue
            status = SCHEDULE_STATUS.get(g["gameStatus"], GameStatus.SCHEDULED)
            has_score = status is not GameStatus.SCHEDULED
            rows.append(
                ScheduledGameRow(
                    game_id=g["gameId"],
                    season=season,
                    season_type=season_type,
                    game_date=date.fromisoformat(g["gameDateEst"][:10]),
                    tip_time_utc=_parse_utc(g.get("gameDateTimeUTC")),
                    home_team_id=home_id,
                    away_team_id=away_id,
                    status=status,
                    home_score=g["homeTeam"]["score"] if has_score else None,
                    away_score=g["awayTeam"]["score"] if has_score else None,
                    is_neutral=bool(g.get("isNeutral")),
                )
            )
    return rows


def _parse_utc(value: str | None) -> datetime | None:
    if not value:
        return None
    try:
        return datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None


def parse_clock(clock: str) -> float:
    """ISO-8601 period clock "PT11M43.00S" -> 703.0 seconds remaining."""
    m = _CLOCK_RE.fullmatch(clock)
    return int(m.group(1)) * 60 + float(m.group(2)) if m else 0.0


def parse_play_by_play(payload: dict[str, Any]) -> list[PbpEventRow]:
    """PlayByPlayV3. Scores are only populated on scoring plays, so they are forward-filled."""
    game_id: str = payload["game"]["gameId"]
    score_home = score_away = 0
    events: dict[int, PbpEventRow] = {}
    for a in payload["game"]["actions"]:
        if a["scoreHome"] != "":
            score_home, score_away = int(a["scoreHome"]), int(a["scoreAway"])
        events[a["actionId"]] = PbpEventRow(
            game_id=game_id,
            action_id=a["actionId"],
            action_number=a["actionNumber"],
            period=a["period"],
            clock_seconds=parse_clock(a["clock"]),
            team_id=a["teamId"] or None,
            location=a["location"] or None,
            person_id=a["personId"] or None,
            action_type=a["actionType"] or None,
            sub_type=a["subType"] or None,
            description=a["description"] or None,
            is_field_goal=bool(a["isFieldGoal"]),
            shot_value=a["shotValue"] or None,
            shot_result=a["shotResult"] or None,
            score_home=score_home,
            score_away=score_away,
        )
    return list(events.values())
