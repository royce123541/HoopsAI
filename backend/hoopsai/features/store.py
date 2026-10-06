import logging
import time
from typing import Any

import numpy as np
import pandas as pd
from sqlalchemy import Engine, insert, text

from hoopsai.db.models import EloRating, GameFeatures
from hoopsai.features.build import FEATURE_VERSION, MODEL_FEATURES, build_features, load_core

log = logging.getLogger(__name__)

_CHUNK = 5000


def _records(frame: pd.DataFrame) -> list[dict[str, Any]]:
    """DataFrame rows -> dicts with NaN as None (JSON null) and numpy scalars as Python."""
    records: list[dict[str, Any]] = (
        frame.astype(object).where(frame.notna(), None).to_dict("records")  # type: ignore[assignment]
    )
    return records


def save_features(engine: Engine, features: pd.DataFrame) -> None:
    """Replace features.game_features and features.elo_ratings in one transaction, so
    readers never see a half-built table."""
    feature_rows = [
        {"game_id": game_id, "feature_version": FEATURE_VERSION, "features": values}
        for game_id, values in zip(
            features["game_id"], _records(features[MODEL_FEATURES]), strict=True
        )
    ]
    elo_rows: list[dict[str, Any]] = []
    for side in ("home", "away"):
        part = features[["game_id", "season", "game_date", f"{side}_team_id"]].copy()
        part.columns = ["game_id", "season", "game_date", "team_id"]
        part["elo_pre"] = features[f"{side}_elo_pre"]
        part["elo_post"] = features[f"{side}_elo_post"].replace({np.nan: None})
        part["game_date"] = part["game_date"].dt.date
        elo_rows.extend(_records(part))

    with engine.begin() as conn:
        conn.execute(text("TRUNCATE features.game_features, features.elo_ratings"))
        for table, rows in ((GameFeatures, feature_rows), (EloRating, elo_rows)):
            for i in range(0, len(rows), _CHUNK):
                conn.execute(insert(table), rows[i : i + _CHUNK])


def rebuild_features(engine: Engine) -> pd.DataFrame:
    started = time.perf_counter()
    games, stats = load_core(engine)
    features = build_features(games, stats)
    save_features(engine, features)
    log.info(
        "built %s features for %d games in %.1fs",
        FEATURE_VERSION,
        len(features),
        time.perf_counter() - started,
    )
    return features
