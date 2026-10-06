from fastapi import APIRouter, HTTPException
from sqlalchemy import select

from hoopsai.api.deps import Conn
from hoopsai.api.schemas import FeatureImportance, ModelInfo
from hoopsai.db.models import ModelVersionSnapshot
from hoopsai.features.labels import feature_label

router = APIRouter(tags=["model"])

TOP_FEATURES = 12


@router.get("/model", response_model=ModelInfo)
async def get_model(conn: Conn) -> ModelInfo:
    """The model behind current predictions: backtest vs Elo, calibration, importance."""
    row = (
        await conn.execute(
            select(ModelVersionSnapshot)
            .order_by(ModelVersionSnapshot.first_used_at.desc())
            .limit(1)
        )
    ).first()
    if row is None:
        raise HTTPException(status_code=404, detail="no model has produced predictions yet")
    m = row._mapping
    backtest = m["backtest"]
    total_gain = sum(m["importance"].values()) or 1.0
    top = sorted(m["importance"].items(), key=lambda kv: -kv[1])[:TOP_FEATURES]
    return ModelInfo(
        version=m["version"],
        run_id=m["run_id"],
        feature_version=m["feature_version"],
        calibration=m["calibration"],
        train_seasons=m["train_seasons"],
        first_used_at=m["first_used_at"],
        seasons=backtest["seasons"],
        pooled_model=backtest["pooled_model"],
        pooled_elo=backtest["pooled_elo"],
        reliability=backtest["reliability"],
        acceptance=backtest["acceptance"],
        top_features=[
            FeatureImportance(
                feature=f, label=feature_label(f, "Home", "Away"), share=gain / total_gain
            )
            for f, gain in top
        ],
    )
