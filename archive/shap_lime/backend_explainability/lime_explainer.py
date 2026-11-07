"""LIME local explanations of single (query, chunk) relevance predictions (tabular, interpretable features)."""
from __future__ import annotations

import numpy as np
from lime.lime_tabular import LimeTabularExplainer


class LimeRelevanceExplainer:
    def __init__(self, predict_proba, feature_names: list[str], background: np.ndarray, seed: int = 42):
        """predict_proba: callable (n, d) -> (n,) probability of the relevant class."""
        self.feature_names = feature_names
        self._p = predict_proba
        self.explainer = LimeTabularExplainer(np.asarray(background, dtype=float), feature_names=feature_names,
                                              class_names=["not relevant", "relevant"], mode="classification",
                                              discretize_continuous=True, random_state=seed)

    def _proba2(self, X):
        p = np.asarray(self._p(np.asarray(X, dtype=float))).reshape(-1)
        return np.column_stack([1 - p, p])

    def explain(self, x: np.ndarray, top: int = 6, num_samples: int = 1500) -> dict:
        x = np.asarray(x, dtype=float).ravel()
        e = self.explainer.explain_instance(x, self._proba2, num_features=top, num_samples=num_samples, labels=(1,))
        return {"method": "LIME (tabular)", "unit": "local linear weight on P(relevant)",
                "local_model_r2": round(float(e.score), 3), "local_prediction": round(float(e.local_pred[0]), 4),
                "model_prediction": round(float(self._p(x.reshape(1, -1))[0]), 4),
                "weights": [{"condition": c, "weight": round(float(w), 4)} for c, w in e.as_list(label=1)],
                "by_feature": {self.feature_names[i]: round(float(w), 4) for i, w in e.as_map()[1]}}
