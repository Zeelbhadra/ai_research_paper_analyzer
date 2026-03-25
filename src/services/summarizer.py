import os
from openai import OpenAI
from src.db import papers_collection

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))


def summarize_text(text):
    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "Summarize research papers concisely."},
                {"role": "user", "content": text}
            ],
            temperature=0.3
        )

        return response.choices[0].message.content

    except Exception as e:
        print("[OpenAI ERROR]", e)
        return None


def summarize_and_store(paper_id, abstract):
    summary = summarize_text(abstract)

    if summary:
        papers_collection.update_one(
            {"paper_id": paper_id},
            {"$set": {"summary": summary}}
        )

    return summary