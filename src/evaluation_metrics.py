"""
Evaluation Metrics for Summarization
Computes ROUGE, BERTScore and Compression Ratio
"""

from rouge_score import rouge_scorer
from bert_score import score


def compute_rouge(reference, generated):

    scorer = rouge_scorer.RougeScorer(
        ["rouge1", "rouge2", "rougeL"],
        use_stemmer=True
    )

    scores = scorer.score(reference, generated)

    return {
        "rouge1": scores["rouge1"].fmeasure,
        "rouge2": scores["rouge2"].fmeasure,
        "rougeL": scores["rougeL"].fmeasure
    }


def compute_bertscore(reference, generated):

    P, R, F1 = score(
        [generated],
        [reference],
        lang="en"
    )

    return F1.mean().item()


def compression_ratio(text, summary):

    return len(summary.split()) / len(text.split())