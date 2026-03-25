import os
import requests
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from src.db import papers_collection
import hashlib
DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data")
os.makedirs(DATA_DIR, exist_ok=True)

executor = ThreadPoolExecutor(max_workers=5)


def _download_task(paper_id, pdf_url):
    if not pdf_url:
        return
    safe_filename = hashlib.md5(paper_id.encode()).hexdigest() + ".pdf"
    file_path = os.path.join(DATA_DIR, f"{paper_id}.pdf")

    if os.path.exists(file_path):
        return
    
    if "arxiv.org/abs" in pdf_url:
        pdf_url = pdf_url.replace("/abs/", "/pdf/") + ".pdf"
    
    try:
        response = requests.get(pdf_url, stream=True, timeout=30)
        response.raise_for_status()

        with open(file_path, "wb") as f:
            for chunk in response.iter_content(8192):
                f.write(chunk)

        papers_collection.update_one(
            {"paper_id": paper_id},
            {
                "$set": {
                    "pdf_cached": True,
                    "local_pdf_path": file_path,
                    "updated_at": datetime.utcnow()
                }
            }
        )

    except Exception as e:
        print("[PDF ERROR]", e)


def cache_paper_pdf_bg(paper_id, pdf_url):
    executor.submit(_download_task, paper_id, pdf_url)