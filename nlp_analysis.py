
from __future__ import annotations
import re
import numpy as np


def clean_concepts(concept_text):
    """Parse comma-separated rubric concepts while preserving short multi-word phrases."""
    return [item.strip().lower() for item in concept_text.split(",") if item.strip()]


def infer_concepts_from_reference(reference_answer, limit=10):
    """Create a transparent fallback rubric from frequent meaningful reference-answer terms."""
    tokens = re.findall(r"[a-zA-Z][a-zA-Z+-]{2,}", reference_answer.lower())
    ignored = {"this", "that", "with", "from", "have", "into", "using", "their", "they", "which", "about", "also", "when"}
    unique = []
    for token in tokens:
        if token not in ignored and token not in unique:
            unique.append(token)
    return unique[:limit]


def _cosine_similarity(vector_a, vector_b):
    denominator = np.linalg.norm(vector_a) * np.linalg.norm(vector_b)
    return float(np.dot(vector_a, vector_b) / denominator) if denominator else 0.0


def evaluate_technical_answer(question, candidate_answer, reference_answer, rubric_concepts="", model_name="all-MiniLM-L6-v2"):
    """Compare an answer with an explicit reference answer and concept rubric.

    A reference answer is deliberately required: embedding similarity without a
    target answer would not be a meaningful correctness measure.
    """
    if not question.strip() or not candidate_answer.strip() or not reference_answer.strip():
        raise ValueError("Question, candidate answer, and reference answer are required for technical evaluation.")
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError as error:
        raise RuntimeError("sentence-transformers is not installed. Run: pip install sentence-transformers") from error

    model = SentenceTransformer(model_name)
    embeddings = model.encode([candidate_answer, reference_answer], normalize_embeddings=True)
    semantic_similarity = round(max(0.0, _cosine_similarity(embeddings[0], embeddings[1])), 3)

    concepts = clean_concepts(rubric_concepts) or infer_concepts_from_reference(reference_answer)
    candidate_lower = candidate_answer.lower()
    covered = [concept for concept in concepts if concept in candidate_lower]
    missing = [concept for concept in concepts if concept not in candidate_lower]
    concept_coverage = round(100 * len(covered) / len(concepts), 1) if concepts else 0.0
    technical_score = round(100 * (0.70 * semantic_similarity + 0.30 * concept_coverage / 100), 1)

    return {
        "question": question,
        "semantic_similarity": semantic_similarity,
        "concept_coverage": concept_coverage,
        "covered_concepts": covered,
        "missing_concepts": missing,
        "rubric_concepts": concepts,
        "technical_score": technical_score,
        "method_note": (
            "Similarity is cosine similarity between pretrained MiniLM sentence embeddings. "
            "It supports consistent rubric-based review; it is not ground truth or an autonomous hiring decision."
        ),
    }
