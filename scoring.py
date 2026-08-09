"""Small, explainable weighted score fusion."""

from __future__ import annotations


def combine_scores(technical_score, communication_score, visual_score):
    """Fuse modality scores using fixed weights chosen for explainability.

    Technical answer quality receives the highest weight because this is a
    technical interview. Communication aids answer delivery; observable visual
    engagement has the lowest weight to avoid overemphasising camera setup.
    """
    overall_score = round(
        0.50 * float(technical_score) + 0.30 * float(communication_score) + 0.20 * float(visual_score), 1
    )
    if overall_score >= 80:
        band = "Strong"
    elif overall_score >= 65:
        band = "Developing"
    else:
        band = "Needs practice"
    return {
        "technical_weight": 0.50,
        "communication_weight": 0.30,
        "visual_weight": 0.20,
        "overall_score": overall_score,
        "performance_band": band,
        "formula": "0.50 × technical + 0.30 × communication + 0.20 × visual",
    }
