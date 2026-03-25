import arxiv
import requests

BASE_URL = "https://api.semanticscholar.org/graph/v1"
SEARCH_FIELDS = "title,abstract,authors,year,citationCount,url,externalIds"


def fetch_arxiv(query: str, max_results: int = 10):
    papers = []
    try:
        client = arxiv.Client()
        search = arxiv.Search(
            query=query,
            max_results=max_results,
            sort_by=arxiv.SortCriterion.Relevance,
        )

        for result in client.results(search):
            papers.append({
                "source": "arxiv",
                "title": result.title,
                "authors": [a.name for a in result.authors],
                "abstract": result.summary,
                "url": result.entry_id,
            })

    except Exception as e:
        print("[arXiv ERROR]", e)

    return papers


def fetch_semantic_scholar(query: str, limit: int = 10):
    papers = []

    params = {
        "query": query,
        "limit": min(limit, 100),
        "fields": SEARCH_FIELDS,
    }

    try:
        resp = requests.get(f"{BASE_URL}/paper/search", params=params, timeout=15)
        resp.raise_for_status()
        data = resp.json()

        for item in data.get("data", []):
            authors = item.get("authors") or []

            papers.append({
                "source": "semantic_scholar",
                "title": item.get("title", ""),
                "authors": [a.get("name", "") for a in authors],
                "abstract": item.get("abstract") or "",
                "url": item.get("url") or "",
            })

    except Exception as e:
        print("[Semantic ERROR]", e)

    return papers


def fetch_all_papers(query: str, limit: int = 10):
    return fetch_arxiv(query, limit) + fetch_semantic_scholar(query, limit)