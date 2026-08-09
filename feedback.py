
from __future__ import annotations
def generate_feedback(video_metrics, audio_metrics, nlp_metrics, final_metrics):

    strengths = []
    weak_areas = []
    suggestions = []

    if nlp_metrics.get("technical_score",0) >= 75:
        strengths.append(
            "The answer aligns well with the supplied technical reference answer."
        )
    else:
        weak_areas.append(
            "Technical alignment with the reference answer needs improvement."
        )


    if nlp_metrics.get("concept_coverage",0) >= 70:
        strengths.append(
            "Most rubric concepts were addressed."
        )
    elif nlp_metrics.get("missing_concepts"):
        missing = ", ".join(
            nlp_metrics["missing_concepts"][:4]
        )
        weak_areas.append(
            f"Missing or unmentioned rubric concepts: {missing}."
        )
        suggestions.append(
            "Use a simple structure: definition, key components, workflow, then a short example."
        )

    wpm = audio_metrics.get(
        "words_per_minute",
        0
    )
    if 120 <= wpm <= 170:
        strengths.append(
            "Speaking pace is within the project target range (120–170 WPM)."
        )

    else:
        weak_areas.append(
            f"Speaking pace is {wpm} WPM, outside the 120–170 WPM target range."
        )
        suggestions.append(
            "Practise answering in 60–90 second blocks and leave brief pauses between ideas."
        )



    if audio_metrics.get("pause_count",0) > max(
        5,
        audio_metrics.get("duration_seconds",0) / 20
    ):
        weak_areas.append(
            "The audio contains many longer silent intervals."
        )
        suggestions.append(
            "Replace filler pauses with a short outline before you begin answering."
        )


    elif audio_metrics.get("clarity_proxy",0) >= 65:
        strengths.append(
            "Audio energy and consistency are good based on the acoustic proxy."
        )


    if video_metrics.get("face_presence",0) >= 85:
        strengths.append(
            "The face remained visible for most sampled frames."
        )

    else:
        weak_areas.append(
            "Face visibility was inconsistent in the sampled video frames."
        )

        suggestions.append(
            "Place the camera at eye level and use stable front lighting."
        )
    if video_metrics.get("eye_contact_score",0) < 55:

        suggestions.append(
            "Keep the camera near your notes and periodically look toward the lens."
        )
    if video_metrics.get("head_stability",0) < 55:

        suggestions.append(
            "Use a stable seated position; natural movement is fine, but avoid frequent large shifts."
        )
    if not suggestions:
        suggestions.append(
            "Keep practising with varied questions and compare each answer against a clear concept rubric."
        )

    return {
        "overall_summary":
            f"{final_metrics.get('performance_band','Unknown')} performance: "
            f"{final_metrics.get('overall_score',0)}/100.",

        "strengths":
            strengths or [
                "No strong signal crossed the current project thresholds."
            ],
        "weak_areas":
            weak_areas or [
                "No major weakness was identified by the selected metrics."
            ],
        "suggestions":
            suggestions,

    }