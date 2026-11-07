# Explainability: explaining the paper, not the model

The goal is that a reader understands **the content of a paper**. Explanations therefore describe the paper, never the retrieval model.

## What exists
- **Plain-language answers**: the assistant is instructed to use simple words, explain technical terms and keep numbers and names exact (`SYSTEM_PROMPT` rule 7, intent notes in `backend/rag/generator.py`).
- **Overview requests** ("explain this paper simply", "why does it matter?") are recognised and answered from the abstract, introduction and conclusion first.
- **Evidence for every statement**: each sentence is tagged *stated* (supported by a cited passage), *interpretation*, or *unverified* ("check this"), using lexical citation verification. Unsupported LLM answers are replaced by the best passages.
- **Highlighted sources**: cited passages are shown with page, section and highlighted supporting words (`backend/explainability/evidence_explainer.py`).
- **One-click explanations** in the Library paper view.

## Removed
SHAP and LIME over a surrogate relevance classifier explained why the *retriever* ranked passages, not what a paper says. The code, routes, demo, tests and charts are preserved in `archive/shap_lime/` and `archive/admin_ui/` (see `archive/README.md`) and are not part of the app; `shap` and `lime` are no longer dependencies.

## Limits
Without an LLM the answers are quoted passages, not simplified rewrites. Simplification quality has not been measured.
