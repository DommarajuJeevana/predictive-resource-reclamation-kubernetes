"""Random Forest forecaster for the near-future *peak* resource requirement."""
from __future__ import annotations
import joblib
import numpy as np
from sklearn.ensemble import RandomForestRegressor

from .config import PolicyConfig
from .data import build_dataset, features_at, feature_names


class PeakForecaster:
    """Predicts max(usage over the next `horizon` readings) from recent history.

    The reclamation decision needs an estimate of how much the workload may *need*
    until the next opportunity to resize, not its average - hence the peak target.
    """

    def __init__(self, cfg: PolicyConfig):
        self.cfg = cfg
        self.model: RandomForestRegressor | None = None
        self.n_train = 0

    @property
    def fitted(self) -> bool:
        return self.model is not None

    def fit(self, hist: np.ndarray) -> bool:
        X, y, _ = build_dataset(np.asarray(hist, float), self.cfg.n_lags, self.cfg.horizon)
        if len(y) < self.cfg.min_train:
            return False
        self.model = RandomForestRegressor(
            n_estimators=self.cfg.n_estimators,
            min_samples_leaf=self.cfg.min_samples_leaf,
            max_features=self.cfg.max_features,
            random_state=self.cfg.seed, n_jobs=1)
        self.model.fit(X, y)
        self.n_train = len(y)
        return True

    def predict(self, hist: np.ndarray) -> float:
        x = features_at(np.asarray(hist, float), self.cfg.n_lags).reshape(1, -1)
        return float(max(self.model.predict(x)[0], 0.0))

    def importances(self) -> dict[str, float]:
        names = feature_names(self.cfg.n_lags)
        return dict(sorted(zip(names, self.model.feature_importances_), key=lambda kv: -kv[1]))

    def save(self, path: str):
        joblib.dump({"cfg": self.cfg, "model": self.model, "n_train": self.n_train}, path)

    @classmethod
    def load(cls, path: str) -> "PeakForecaster":
        d = joblib.load(path)
        obj = cls(d["cfg"]); obj.model = d["model"]; obj.n_train = d["n_train"]
        return obj


def walk_forward(series: np.ndarray, cfg: PolicyConfig) -> "pd.DataFrame":
    """Honest out-of-sample evaluation: at every reading t >= warmup, fit on windows
    that were *complete by t*, forecast the next-`horizon` peak, compare to truth."""
    import pandas as pd
    series = np.asarray(series, float)
    rows = []
    fc = PeakForecaster(cfg)
    for t in range(cfg.warmup, len(series) - cfg.horizon):
        if (t - cfg.warmup) % cfg.retrain_every == 0:
            fc.fit(series[: t + 1])
        if not fc.fitted:
            continue
        h = series[: t + 1]
        rows.append(dict(
            t=t,
            actual=series[t + 1: t + 1 + cfg.horizon].max(),
            rf=fc.predict(h),
            last=h[-1],
            max4=h[-4:].max(),
            max8=h[-8:].max(),
            mean_hist=h.mean(),
            p90_hist=np.percentile(h, 90),
        ))
    return pd.DataFrame(rows)


def forecast_metrics(wf: "pd.DataFrame") -> "pd.DataFrame":
    import pandas as pd
    out = []
    for m in ["rf", "last", "max4", "max8", "mean_hist", "p90_hist"]:
        e = wf[m] - wf["actual"]
        out.append(dict(
            predictor=m,
            MAE=e.abs().mean(),
            RMSE=float(np.sqrt((e ** 2).mean())),
            under_rate=(e < 0).mean(),                    # predicted less than needed
            mean_shortfall=(-e[e < 0]).mean() if (e < 0).any() else 0.0,
        ))
    return pd.DataFrame(out).set_index("predictor")
