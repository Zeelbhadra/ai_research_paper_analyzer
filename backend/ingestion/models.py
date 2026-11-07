"""Source-agnostic paper metadata record, deduplication keys and shared HTTP helper."""
from __future__ import annotations

import logging
import re
import time
from dataclasses import asdict, dataclass, field

import requests

log = logging.getLogger(__name__)

_ARXIV_ID = re.compile(r"(?:arxiv\.org/(?:abs|pdf)/|arxiv:)?((?:\d{4}\.\d{4,5})|(?:[a-z\-]+(?:\.[A-Z]{2})?/\d{7}))(?:v\d+)?(?:\.pdf)?", re.I)


def normalize_title(title: str) -> str:
    return re.sub(r"\s+", " ", re.sub(r"[^a-z0-9]+", " ", (title or "").lower())).strip()


def normalize_doi(doi: str | None) -> str | None:
    if not doi:
        return None
    doi = re.sub(r"^(https?://(dx\.)?doi\.org/|doi:)", "", doi.strip(), flags=re.I)
    return doi.lower() or None


def extract_arxiv_id(value: str | None) -> str | None:
    """Return the version-less arXiv id from an id / abs URL / pdf URL, else None."""
    if not value:
        return None
    m = _ARXIV_ID.search(value.strip())
    return m.group(1).lower() if m else None


@dataclass
class PaperRecord:
    title: str
    source: str                          # 'arxiv' | 'semantic_scholar' | 'upload' | ...
    abstract: str = ""
    authors: list[str] = field(default_factory=list)
    published: str | None = None         # ISO date or year
    category: str | None = None
    doi: str | None = None
    arxiv_id: str | None = None
    abs_url: str | None = None
    pdf_url: str | None = None
    citation_count: int | None = None
    venue: str | None = None
    paper_id: str = ""                   # stable internal id, assigned by Deduplicator/store
    pdf_path: str | None = None
    status: str = "discovered"           # discovered|downloaded|processed|failed
    sources_seen: list[str] = field(default_factory=list)
    enrichment: dict = field(default_factory=dict)   # keywords / summary / topics added by the document agent

    def __post_init__(self):
        self.doi = normalize_doi(self.doi)
        self.arxiv_id = extract_arxiv_id(self.arxiv_id) if self.arxiv_id else extract_arxiv_id(self.abs_url)
        if not self.sources_seen:
            self.sources_seen = [self.source]

    def to_dict(self) -> dict:
        return asdict(self)


class HttpError(Exception):
    pass


def request_with_retry(method: str, url: str, *, session: requests.Session | None = None,
                       max_retries: int = 4, timeout: int = 20, backoff: float = 1.5,
                       sleep=time.sleep, **kw) -> requests.Response:
    """HTTP with exponential backoff; honours Retry-After on 429/503; retries 5xx/network errors.
    4xx other than 429 fail immediately."""
    s = session or requests
    last: Exception | None = None
    for attempt in range(max_retries + 1):
        try:
            r = s.request(method, url, timeout=timeout, **kw)
            if r.status_code in (429, 500, 502, 503, 504):
                ra = r.headers.get("Retry-After")
                delay = float(ra) if ra and ra.replace(".", "", 1).isdigit() else backoff * (2 ** attempt)
                last = HttpError(f"{r.status_code} from {url}")
                if attempt < max_retries:
                    log.warning("retry %d/%d for %s (%s), sleeping %.1fs", attempt + 1, max_retries, url, r.status_code, delay)
                    sleep(min(delay, 60))
                continue
            r.raise_for_status()
            return r
        except (requests.ConnectionError, requests.Timeout) as e:
            last = e
            if attempt < max_retries:
                sleep(min(backoff * (2 ** attempt), 60))
        except requests.HTTPError as e:
            raise HttpError(str(e)) from e
    raise HttpError(f"giving up on {url}: {last}")
