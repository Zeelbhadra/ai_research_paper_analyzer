"""
Text Processor
Cleans and segments raw paper text for downstream NLP tasks.
"""

import re


def clean_text(raw: str) -> str:
    """
    Clean raw extracted text:
    - Collapse whitespace
    - Remove page numbers / headers
    - Strip reference section
    """
    # Remove common page-number patterns like "Page 3 of 12", standalone numbers
    text = re.sub(r"(?m)^[\s]*\d+[\s]*$", "", raw)

    # Remove email addresses
    text = re.sub(r"\S+@\S+\.\S+", "", text)

    # Collapse multiple whitespace / newlines
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = re.sub(r"[ \t]+", " ", text)

    # Strip everything after "References" or "Bibliography"
    text = re.split(
        r"\n\s*(?:References|REFERENCES|Bibliography|BIBLIOGRAPHY)\s*\n",
        text,
        maxsplit=1,
    )[0]

    return text.strip()


def segment_sections(text: str) -> dict[str, str]:
    """
    Attempt to split a paper into named sections using heading heuristics.

    Returns dict with keys like 'abstract', 'introduction', 'methodology',
    'results', 'conclusion', and 'full_text' (always present).
    """
    sections = {"full_text": text}

    # Common section heading patterns (numbered or unnumbered)
    heading_pattern = re.compile(
        r"\n\s*(?:\d+\.?\s+)?"
        r"(Abstract|Introduction|Background|Related\s+Work|"
        r"Methodology|Methods?|Approach|"
        r"Experiments?|Results?|Evaluation|"
        r"Discussion|Conclusion|Summary)"
        r"\s*\n",
        re.IGNORECASE,
    )

    matches = list(heading_pattern.finditer(text))

    for i, match in enumerate(matches):
        section_name = re.sub(r"\s+", "_", match.group(1).strip().lower())
        start = match.end()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        content = text[start:end].strip()
        if content:
            sections[section_name] = content

    return sections


def truncate_for_model(text: str, max_chars: int = 4000) -> str:
    """Truncate text to roughly fit within model token limits."""
    if len(text) <= max_chars:
        return text
    return text[:max_chars]
