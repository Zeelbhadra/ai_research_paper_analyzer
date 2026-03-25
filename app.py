import os
import traceback

from flask import Flask, jsonify, request, render_template

from src.services.paper_fetcher import fetch_all_papers
from src.services.paper_service import get_summary, save_summary
from src.services.summarizer import summarize_text

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 16 * 1024 * 1024


@app.route("/")
def index():
    return render_template("index.html")


# SEARCH + SUMMARY API
@app.route("/api/search")
def api_search():
    query = request.args.get("q", "").strip()
    limit = min(int(request.args.get("limit", 10)), 20)

    if not query:
        return jsonify({"error": "Missing query"}), 400

    try:
        papers = fetch_all_papers(query, limit)

        results = []

        for p in papers:
            paper_id = p.get("url")

            # Check if summary exists
            summary = get_summary(paper_id)

            # If not, generate + store
            if not summary and p.get("abstract"):
                summary = summarize_text(p["abstract"])
                if summary:
                    save_summary(paper_id, p, summary)

            results.append({
                "title": p.get("title"),
                "authors": p.get("authors"),
                "url": p.get("url"),
                "source": p.get("source"),
                "summary": summary or "Summary not available"
            })

        return jsonify({"results": results})

    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)