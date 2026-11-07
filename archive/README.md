# Archive

Code removed from the product but kept for reference (not imported, not tested by default).

- `shap_lime/`: SHAP/LIME explanations of the retrieval relevance model (feature extractor, relevance classifier, SHAP and LIME explainers,
  retrieval explainer, its API route and demo, and the original combined test file). Removed because it explains the *model*, not the
  *content of the papers*. To revive: copy `backend_explainability/*` back into `backend/explainability/`, `routes_explainability.py` to
  `backend/api/routes/explainability.py`, and add `shap` and `lime` to requirements.
- `admin_ui/`: the Dashboard, Benchmarks, Explainability and Pipeline pages (JavaScript) and the chart library. They need the admin API
  (`ADMIN_API=1`) and show model/backend details, so they are not part of the reader-facing UI.
- `legacy_ui/`: the original single-page script, replaced by `static/js/view_discover.js`.
