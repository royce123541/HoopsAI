"""HTTP client for ESPN's public NBA JSON (site.api.espn.com): no key, and unlike stats.nba.com
it answers cloud servers such as GitHub Actions runners."""

import json
import urllib.request
from datetime import date
from typing import Any

from hoopsai.sources.base import RawResponse
from hoopsai.sources.throttle import Throttle, call_with_retry

BASE_URL = "https://site.api.espn.com/apis/site/v2/sports/basketball/nba"
SOURCE = "espn"


class EspnClient:
    name = SOURCE

    def __init__(
        self,
        min_interval_s: float = 0.3,
        attempts: int = 4,
        base_delay_s: float = 2.0,
        timeout_s: int = 20,
    ) -> None:
        self._throttle = Throttle(min_interval_s)
        self._attempts = attempts
        self._base_delay_s = base_delay_s
        self._timeout_s = timeout_s

    def _get(self, endpoint: str, params: dict[str, Any]) -> RawResponse:
        query = "&".join(f"{k}={v}" for k, v in params.items())
        url = f"{BASE_URL}/{endpoint}?{query}"

        def attempt() -> dict[str, Any]:
            self._throttle.wait()
            req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
            with urllib.request.urlopen(req, timeout=self._timeout_s) as resp:
                payload: dict[str, Any] = json.load(resp)
            return payload

        payload = call_with_retry(
            attempt, attempts=self._attempts, base_delay_s=self._base_delay_s, what=url
        )
        return RawResponse(source=SOURCE, endpoint=endpoint, params=params, payload=payload)

    def fetch_scoreboard_day(self, day: date) -> RawResponse:
        return self._get("scoreboard", {"dates": f"{day:%Y%m%d}"})

    def fetch_scoreboard_month(self, year: int, month: int) -> RawResponse:
        """Every game of a month in one response (ESPN rejects explicit date ranges)."""
        return self._get("scoreboard", {"dates": f"{year}{month:02d}", "limit": 1000})

    def fetch_summary(self, espn_id: str) -> RawResponse:
        """Box score, header and full play-by-play of one game."""
        return self._get("summary", {"event": espn_id})
