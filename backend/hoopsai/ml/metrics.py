import numpy as np
import numpy.typing as npt

PROB_EPS = 1e-3  # predictions are clipped to [eps, 1 - eps] so log loss stays finite

FloatArray = npt.NDArray[np.float64]


def clip(p: npt.ArrayLike) -> FloatArray:
    clipped: FloatArray = np.clip(np.asarray(p, dtype=float), PROB_EPS, 1 - PROB_EPS)
    return clipped


def log_loss(y: npt.ArrayLike, p: npt.ArrayLike) -> float:
    y_, p_ = np.asarray(y, dtype=float), clip(p)
    return float(-np.mean(y_ * np.log(p_) + (1 - y_) * np.log(1 - p_)))


def brier(y: npt.ArrayLike, p: npt.ArrayLike) -> float:
    return float(np.mean((np.asarray(p, dtype=float) - np.asarray(y, dtype=float)) ** 2))


def accuracy(y: npt.ArrayLike, p: npt.ArrayLike) -> float:
    return float(np.mean((np.asarray(p) >= 0.5) == (np.asarray(y) == 1)))


def reliability(y: npt.ArrayLike, p: npt.ArrayLike, bins: int = 10) -> list[dict[str, float]]:
    """Equal-width probability bins: mean prediction vs observed home-win rate."""
    y_, p_ = np.asarray(y, dtype=float), np.asarray(p, dtype=float)
    idx = np.minimum((p_ * bins).astype(int), bins - 1)
    out = []
    for b in range(bins):
        mask = idx == b
        if mask.any():
            out.append(
                {
                    "bin_low": b / bins,
                    "bin_high": (b + 1) / bins,
                    "mean_pred": float(p_[mask].mean()),
                    "observed": float(y_[mask].mean()),
                    "count": int(mask.sum()),
                }
            )
    return out


def ece(y: npt.ArrayLike, p: npt.ArrayLike, bins: int = 10) -> float:
    """Expected calibration error: count-weighted |mean prediction - observed rate|."""
    curve = reliability(y, p, bins)
    total = sum(b["count"] for b in curve)
    return float(sum(b["count"] * abs(b["mean_pred"] - b["observed"]) for b in curve) / total)


def summary(y: npt.ArrayLike, p: npt.ArrayLike) -> dict[str, float]:
    return {
        "logloss": log_loss(y, p),
        "brier": brier(y, p),
        "accuracy": accuracy(y, p),
        "ece": ece(y, p),
        "n": float(len(np.asarray(y))),
    }
