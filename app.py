import logging
import os
import threading

from flask import Flask, render_template

from backend.monitoring.logger import setup_logging

setup_logging(os.getenv("LOG_LEVEL", "INFO"), json_logs=os.getenv("LOG_FORMAT", "text") == "json")


def create_app(admin: bool | None = None) -> Flask:
    """Public routes (what readers use) are always registered. The diagnostic/admin routes (dashboard, benchmarks, pipeline control,
    retrieval status, raw RAG responses) expose backend details and are registered only with admin=True / ADMIN_API=1."""
    admin = (os.getenv("ADMIN_API") == "1") if admin is None else admin
    app = Flask(__name__)
    app.config["MAX_CONTENT_LENGTH"] = 16 * 1024 * 1024
    app.json.sort_keys = False      # keep ranked dicts in the order the server built them
    app.config["ADMIN_API"] = admin

    from backend.api.routes.chat import bp as chat_bp
    from backend.api.routes.health import bp as health_bp
    from backend.api.routes.legacy import bp as discover_bp
    from backend.api.routes.library import bp as library_bp
    for bp in (discover_bp, library_bp, chat_bp, health_bp):
        app.register_blueprint(bp)
    if admin:
        from backend.api.routes.dashboard import bp as dashboard_bp
        from backend.api.routes.pipeline import bp as pipeline_bp
        from backend.api.routes.rag import bp as rag_bp
        from backend.api.routes.retrieval import bp as retrieval_bp
        for bp in (dashboard_bp, retrieval_bp, rag_bp, pipeline_bp):
            app.register_blueprint(bp)

    @app.route("/")
    def index():
        return render_template("index.html")

    return app


app = create_app()


def _warm_up():
    """WARMUP=1: load the retrieval service and the pipeline system in the background so the first request is fast."""
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
