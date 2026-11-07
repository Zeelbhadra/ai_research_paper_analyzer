import logging
import os
import threading

from flask import Flask, render_template

from backend.monitoring.logger import setup_logging

setup_logging(os.getenv("LOG_LEVEL", "INFO"), json_logs=os.getenv("LOG_FORMAT", "text") == "json")

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 16 * 1024 * 1024
app.json.sort_keys = False      # keep ranked dicts (e.g. feature importances) in the order the server built them

from backend.api.routes.legacy import bp as legacy_bp  # noqa: E402  original PaperLens endpoints (fixed)
from backend.api.routes.library import bp as library_bp  # noqa: E402
from backend.api.routes.dashboard import bp as dashboard_bp  # noqa: E402
from backend.api.routes.retrieval import bp as retrieval_bp  # noqa: E402
from backend.api.routes.rag import bp as rag_bp  # noqa: E402
from backend.api.routes.explainability import bp as explain_bp  # noqa: E402
from backend.api.routes.pipeline import bp as pipeline_bp  # noqa: E402
from backend.api.routes.health import bp as health_bp  # noqa: E402

for _bp in (legacy_bp, library_bp, dashboard_bp, retrieval_bp, rag_bp, explain_bp, pipeline_bp, health_bp):
    app.register_blueprint(_bp)


@app.route("/")
def index():
    return render_template("index.html")


def _warm_up():
    """WARMUP=1: load the retrieval service and the pipeline system in the background so the first user request is fast."""
    log = logging.getLogger("warmup")
    try:
        from backend.api.routes.pipeline import get_system
        from backend.api.routes.retrieval import get_production
        get_production()
        get_system()
        log.info("warm-up complete")
    except Exception:
        log.exception("warm-up failed (the service will load lazily on first use)")


if os.getenv("WARMUP") == "1":
    threading.Thread(target=_warm_up, name="warmup", daemon=True).start()

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
