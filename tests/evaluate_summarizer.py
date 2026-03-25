import numpy as np
from tqdm import tqdm

import os
import requests

from src.arxiv_fetcher import search_papers
from src.pdf_extractor import extract_text
from src.summarizer import generate_summary
from src.evaluation_metrics import (
    compute_rouge,
    compute_bertscore,
    compression_ratio
)

QUERY = "machine learning"
NUM_PAPERS = 20

def download_pdf(title, pdf_url):

    if not pdf_url:
        return None

    os.makedirs("data", exist_ok=True)

    safe_title = "".join(
        c for c in title if c.isalnum() or c in (" ", "-", "_")
    ).strip()[:80]

    file_path = os.path.join("data", safe_title + ".pdf")

    if os.path.exists(file_path):
        return file_path

    try:

        headers = {
            "User-Agent": "Mozilla/5.0 Research Paper Analyzer"
        }

        r = requests.get(pdf_url, headers=headers, timeout=30)

        r.raise_for_status()

        with open(file_path, "wb") as f:
            f.write(r.content)

        return file_path

    except Exception as e:

        print("Download failed:", e)

        return None

def evaluate():

    papers = search_papers(QUERY, max_results=NUM_PAPERS)

    rouge1_scores = []
    rouge2_scores = []
    rougeL_scores = []
    bert_scores = []
    compression_scores = []

    valid_papers = 0

    for paper in tqdm(papers):

        pdf_url = paper.get("pdf_url")

        if not pdf_url:
            continue

        try:

            # Download paper
            pdf_path = download_pdf(paper["title"], pdf_url)

            if not pdf_path:
                continue

            # Extract full text
            full_text = extract_text(pdf_path)

            if len(full_text) < 1000:
                continue

            reference = paper["abstract"]

            # Generate summary
            generated = generate_summary(full_text)

            # ROUGE
            rouge = compute_rouge(reference, generated)

            rouge1_scores.append(rouge["rouge1"])
            rouge2_scores.append(rouge["rouge2"])
            rougeL_scores.append(rouge["rougeL"])

            # BERTScore
            bert = compute_bertscore(reference, generated)
            bert_scores.append(bert)

            # Compression
            compression = compression_ratio(full_text, generated)
            compression_scores.append(compression)

            valid_papers += 1

        except Exception as e:

            print("Skipping paper due to error:", e)

            continue

    print("\n===== Evaluation Results =====\n")

    print("Papers evaluated:", valid_papers)

    print("\nAverage ROUGE-1:", np.mean(rouge1_scores))
    print("Average ROUGE-2:", np.mean(rouge2_scores))
    print("Average ROUGE-L:", np.mean(rougeL_scores))

    print("\nAverage BERTScore:", np.mean(bert_scores))

    print("\nAverage Compression Ratio:", np.mean(compression_scores))


if __name__ == "__main__":
    evaluate()