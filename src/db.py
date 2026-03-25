import os
from pymongo import MongoClient
from dotenv import load_dotenv

load_dotenv()

MONGO_URI = os.getenv("MONGO_URI")

try:
    client = MongoClient(MONGO_URI, serverSelectionTimeoutMS=5000)
    client.admin.command("ping")

    db = client["ai_paper_analyzer"]
    summaries_collection = db["summaries"]

    summaries_collection.create_index("paper_id", unique=True)

    print("[MongoDB] Connected")

except Exception as e:
    print("[MongoDB ERROR]", e)
    summaries_collection = None