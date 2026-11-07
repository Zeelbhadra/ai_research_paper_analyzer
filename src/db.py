import os
from pymongo import MongoClient
from dotenv import load_dotenv

load_dotenv()

MONGO_URI = os.getenv("MONGO_URI")

summaries_collection = None
papers_collection = None

try:
    if not MONGO_URI:               # optional legacy cache: do not wait on a connection that cannot succeed
        raise RuntimeError("MONGO_URI not set; summary cache disabled")
    client = MongoClient(MONGO_URI, serverSelectionTimeoutMS=5000)
    client.admin.command("ping")

    db = client["ai_paper_analyzer"]
    summaries_collection = db["summaries"]
    papers_collection = db["papers"]

    summaries_collection.create_index("paper_id", unique=True)

    print("[MongoDB] Connected")

except Exception as e:
    print("[MongoDB ERROR]", e)
    summaries_collection = None
    papers_collection = None