"""Streamlit UI for the multimodal interview practice evaluator."""

from __future__ import annotations

import tempfile
from pathlib import Path

import streamlit as st

from audio_analysis import (
    analyze_audio,
    extract_audio_from_video,
    transcribe_audio,
)

from feedback import generate_feedback
from nlp_analysis import evaluate_technical_answer
from report import (
    build_report,
    save_json_report,
    save_pdf_report,
)

from scoring import combine_scores
from video_analysis import analyze_video


BASE_DIRECTORY = Path(__file__).parent
REPORT_DIRECTORY = BASE_DIRECTORY / "reports"



def show_metric_columns(video_metrics, audio_metrics, nlp_metrics, final_metrics):

    st.subheader("Evaluation summary")

    columns = st.columns(4)

    columns[0].metric(
        "Visual",
        f"{video_metrics['visual_score']}/100"
    )

    columns[1].metric(
        "Communication",
        f"{audio_metrics['communication_score']}/100"
    )

    columns[2].metric(
        "Technical",
        f"{nlp_metrics['technical_score']}/100"
    )

    columns[3].metric(
        "Overall",
        f"{final_metrics['overall_score']}/100"
    )



def show_details(video_metrics, audio_metrics, nlp_metrics, feedback):

    visual_tab, audio_tab, technical_tab, feedback_tab = st.tabs(
        [
            "Video",
            "Audio",
            "Technical",
            "Feedback"
        ]
    )

    with visual_tab:
        st.json(
            {
                key:value
                for key,value in video_metrics.items()
                if key!="method_note"
            }
        )
        st.caption(
            video_metrics["method_note"]
        )
    with audio_tab:
        st.metric(
            "Speaking Rate",
            f"{audio_metrics['words_per_minute']} WPM"
        )
        st.metric(
            "Pauses",
            audio_metrics["pause_count"]
        )
        st.metric(
            "Clarity Proxy",
            f"{audio_metrics['clarity_proxy']}/100"
        )
        st.text_area(
            "Transcript",
            audio_metrics["transcript"],
            height=180,
            disabled=True
        )

    with technical_tab:
        st.metric(
            "Semantic Similarity",
            nlp_metrics["semantic_similarity"]
        )
        st.metric(
            "Concept Coverage",
            f"{nlp_metrics['concept_coverage']}%"
        )
        st.write(
            "Covered Concepts:",
            ", ".join(
                nlp_metrics["covered_concepts"]
            )
            or "None"
        )
        st.write(
            "Missing Concepts:",
            ", ".join(
                nlp_metrics["missing_concepts"]
            )
            or "None"
        )

    with feedback_tab:
        st.write(
            feedback["overall_summary"]
        )
        st.write("### Strengths")
        for item in feedback["strengths"]:
            st.write("-", item)
        st.write("### Improvements")
        for item in feedback["weak_areas"]:
            st.write("-", item)
        st.write("### Suggestions")
        for item in feedback["suggestions"]:
            st.write("-", item)

def main():
    st.set_page_config(
        page_title="AI Interview Evaluator",
        page_icon="🎙️",
        layout="wide"
    )
    st.title(
        "Multimodal AI Interview Evaluation System"
    )
    st.caption(
        "Video + Audio + NLP based interview analysis"
    )
    with st.sidebar:
        whisper_model = st.selectbox(
            "Whisper Model",
            [
                "tiny",
                "base",
                "small"
            ],
            index=1
        )
    uploaded_video = st.file_uploader(
        "Upload Interview Video",
        type=[
            "mp4",
            "mov",
            "avi",
            "mkv"
        ]
    )
    question = st.text_input(
        "Interview Question"
    )
    reference_answer = st.text_area(
        "Reference Answer / Expected Answer",
        height=130
    )
    rubric_concepts = st.text_input(
        "Expected Concepts (optional)"
    )
    typed_answer = st.text_area(
        "Candidate Answer Override (optional)",
        height=120
    )
    if st.button(
        "Analyze Interview",
        type="primary"
    ):
        if uploaded_video is None and not typed_answer.strip():
            st.error(
                "Upload video or enter candidate answer."
            )
            return
        if not question.strip() or not reference_answer.strip():

            st.error(
                "Question and reference answer required."
            )
            return
        if uploaded_video is None:

            st.info(
                "Running text-only NLP evaluation mode."
            )

        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            audio_path = temp_path / "audio.wav"
            video_metrics = {
                "duration_seconds":0,
                "sampled_frames":0,
                "face_frames":0,
                "face_presence":0,
                "eye_contact_score":0,
                "head_stability":0,
                "visual_score":0,
                "method_note":
                "Video not provided"

            }
            audio_metrics = {
                "duration_seconds":0,
                "word_count":0,
                "words_per_minute":0,
                "pause_count":0,
                "clarity_proxy":0,
                "communication_score":0,
                "transcript":typed_answer,
                "method_note":
                "Manual answer used"

            }



            try:
                transcript = ""
                if uploaded_video is not None:
                    video_path = (
                        temp_path /
                        uploaded_video.name
                    )
                    video_path.write_bytes(
                        uploaded_video.getbuffer()
                    )
                    with st.spinner(
                        "Analyzing video..."
                    ):

                        video_metrics = analyze_video(
                            video_path
                        )
                    with st.spinner(
                        "Transcribing audio..."
                    ):
                        extract_audio_from_video(
                            video_path,
                            audio_path
                        )
                        transcript = transcribe_audio(
                            audio_path,
                            whisper_model
                        )
                        audio_metrics = analyze_audio(
                            audio_path,
                            transcript
                        )
                if typed_answer.strip():
                    candidate_answer = typed_answer.strip()
                else:
                    candidate_answer = transcript
                if not candidate_answer:
                    raise ValueError(
                        "No candidate answer found."
                    )

                with st.spinner(
                    "Evaluating technical answer..."
                ):
                    nlp_metrics = evaluate_technical_answer(
                        question,
                        candidate_answer,
                        reference_answer,
                        rubric_concepts
                    )

                final_metrics = combine_scores(
                    nlp_metrics["technical_score"],
                    audio_metrics["communication_score"],
                    video_metrics["visual_score"]

                )
                feedback = generate_feedback(
                    video_metrics,
                    audio_metrics,
                    nlp_metrics,
                    final_metrics

                )

                report_data = build_report(
                    question,
                    video_metrics,
                    audio_metrics,
                    nlp_metrics,
                    final_metrics,
                    feedback

                )
                json_path = save_json_report(
                    report_data,
                    REPORT_DIRECTORY
                )
                pdf_path = save_pdf_report(
                    report_data,
                    REPORT_DIRECTORY
                )

            except Exception as error:
                st.exception(error)
                return

        show_metric_columns(
            video_metrics,
            audio_metrics,
            nlp_metrics,
            final_metrics
        )


        show_details(
            video_metrics,
            audio_metrics,
            nlp_metrics,
            feedback
        )


        st.success(
            "Analysis completed successfully"
        )


        col1,col2 = st.columns(2)
        col1.download_button(
            "Download JSON Report",
            json_path.read_bytes(),
            json_path.name,
            "application/json"
        )
        col2.download_button(
            "Download PDF Report",
            pdf_path.read_bytes(),
            pdf_path.name,
            "application/pdf"
        )

if __name__ == "__main__":

    main()