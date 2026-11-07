"""SHAP attributions for the relevance model (linear or tree), in log-odds of 'relevant'.

SHAP here explains the relevance model's prediction from interpretable features. It does NOT explain
embedding dimensions or the semantic content of the dense model.
"""
from __future__ import annotations

import numpy as np
import shap
from sklearn.pipeline import Pipeline


class ShapRelevanceExplainer:
    def __init__(self, model, feature_names: list[str], background: np.ndarray):
        self.feature_names = feature_names
        self.background = np.asarray(background, dtype=float)
        self._scaler = None
        est = model
        if isinstance(model, Pipeline):                      # scaler -> linear model: explain in scaled space (same attribution per feature)
            self._scaler, est = model[0], model[-1]
        bg = self._scaler.transform(self.background) if self._scaler is not None else self.background
        if hasattr(est, "coef_"):
            self.explainer = shap.LinearExplainer(est, bg)
            self.kind = "linear"
        else:
            self.explainer = shap.TreeExplainer(est, data=bg, feature_perturbation="interventional", model_output="raw")
            self.kind = "tree"
        self.model = model

    def _xs(self, X):
        X = np.atleast_2d(np.asarray(X, dtype=float))
        return self._scaler.transform(X) if self._scaler is not None else X

    def shap_values(self, X) -> np.ndarray:
        v = self.explainer.shap_values(self._xs(X))
        if isinstance(v, list):                              # older shap: per-class list
            v = v[-1]
        v = np.asarray(v)
        return v[..., -1] if v.ndim == 3 else v

    def base_value(self) -> float:
        b = np.asarray(self.explainer.expected_value)
        return float(b.ravel()[-1])

    def explain(self, x: np.ndarray, top: int = 6) -> dict:
        x = np.asarray(x, dtype=float).ravel()
        sv = self.shap_values(x)[0]
        base = self.base_value()
        logit = float(self.model.decision_function(x.reshape(1, -1))[0]) if hasattr(self.model, "decision_function") else None
        order = np.argsort(-np.abs(sv))[:top]
        return {"method": f"SHAP ({self.kind})", "unit": "log-odds of relevance", "base_value": round(base, 4),
                "prediction_logodds": round(base + float(sv.sum()), 4),
                "additivity_error": None if logit is None else round(abs(base + float(sv.sum()) - logit), 6),
                "contributions": [{"feature": self.feature_names[i], "value": round(float(x[i]), 4),
                                   "shap": round(float(sv[i]), 4)} for i in order]}

    def global_importance(self, X) -> dict[str, float]:
        m = np.abs(self.shap_values(X)).mean(0)
        return dict(sorted(zip(self.feature_names, (round(float(v), 4) for v in m)), key=lambda t: -t[1]))
