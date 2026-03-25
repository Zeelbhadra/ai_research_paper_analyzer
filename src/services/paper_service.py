from src.db import summaries_collection
from datetime import datetime


def get_summary(paper_id):
    if summaries_collection is None:
        return None

    doc = summaries_collection.find_one({"paper_id": paper_id})
    return doc.get("summary") if doc else None


def save_summary(paper_id, paper, summary):
    if summaries_collection is None:
        return

    doc = {
        "paper_id": paper_id,
        "title": paper.get("title"),
        "authors": paper.get("authors"),
        "source": paper.get("source"),
        "summary": summary,
        "created_at": datetime.utcnow()
    }

    try:
        summaries_collection.update_one(
            {"paper_id": paper_id},
            {"$set": doc},
            upsert=True
        )
        print(f"[DB] Saved summary: {paper.get('title')}")
    except Exception as e:
        print("[DB ERROR]", e)