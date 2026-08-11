
from __future__ import annotations
import json
from datetime import datetime
from pathlib import Path


def build_interview_report(topic, question_results, feedback, resume_name=""):
    """Create one report for a complete multi-question live interview."""
    return {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "interview_topic": topic,
        "resume_filename": resume_name or None,
        "overall_technical_score": feedback["overall_technical_score"],
        "question_wise_results": question_results,
        "strengths": feedback["strengths"],
        "weak_areas": feedback["weak_areas"],
        "final_feedback": feedback["final_feedback"],
        "ethical_note": "This educational project provides practice feedback only. Do not use it as an automated hiring decision tool.",
    }


def build_report(question, video_metrics, audio_metrics, nlp_metrics, final_metrics, feedback):
    return {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "question": question,
        "visual_analysis": video_metrics,
        "audio_analysis": audio_metrics,
        "technical_analysis": nlp_metrics,
        "final_score": final_metrics,
        "feedback": feedback,
        "ethical_note": "This educational project provides practice feedback only. Do not use it as an automated hiring decision tool.",
    }


def save_json_report(report_data, output_directory):
    output_path = Path(output_directory)
    output_path.mkdir(parents=True, exist_ok=True)
    filename = f"interview_report_{datetime.now():%Y%m%d_%H%M%S}.json"
    file_path = output_path / filename
    file_path.write_text(json.dumps(report_data, indent=2, ensure_ascii=False), encoding="utf-8")
    return file_path


def _write_wrapped_line(pdf, text, x, y, max_chars=92):
    """Write simple wrapped text and return the next y coordinate."""
    words = str(text).split()
    line = ""
    for word in words:
        if len(line) + len(word) + 1 > max_chars:
            pdf.drawString(x, y, line)
            y -= 14
            line = word
        else:
            line = f"{line} {word}".strip()
    if line:
        pdf.drawString(x, y, line)
        y -= 14
    return y


def save_pdf_report(report_data, output_directory):
    """Create a concise, readable PDF summary using reportlab's procedural canvas API."""
    try:
        from reportlab.lib.pagesizes import A4
        from reportlab.pdfgen import canvas
    except ImportError as error:
        raise RuntimeError("reportlab is not installed. Run: pip install reportlab") from error

    output_path = Path(output_directory)
    output_path.mkdir(parents=True, exist_ok=True)
    file_path = output_path / f"interview_report_{datetime.now():%Y%m%d_%H%M%S}.pdf"
    pdf = canvas.Canvas(str(file_path), pagesize=A4)
    width, height = A4
    y = height - 48

    def new_page_if_needed(current_y):
        if current_y < 52:
            pdf.showPage()
            return height - 48
        return current_y

    pdf.setFont("Helvetica-Bold", 16)
    pdf.drawString(42, y, "AI Interview Practice Report")
    y -= 28
    pdf.setFont("Helvetica", 10)
    if "question_wise_results" in report_data:
        y = _write_wrapped_line(pdf, f"Topic: {report_data['interview_topic']}", 42, y)
        if report_data.get("resume_filename"):
            y = _write_wrapped_line(pdf, f"Resume used: {report_data['resume_filename']}", 42, y)
        y = _write_wrapped_line(
            pdf,
            f"Overall technical score: {report_data['overall_technical_score']}/10",
            42,
            y,
        )
        y -= 8
        sections = []
        for index, item in enumerate(report_data["question_wise_results"], start=1):
            analysis = item["technical_analysis"]
            sections.append((
                f"Question {index}: {analysis['question']}",
                [
                    f"Score: {analysis['technical_score']}/10 | Difficulty: {analysis['difficulty']}",
                    f"Covered: {', '.join(analysis['covered_points']) or 'None'}",
                    f"Missing: {', '.join(analysis['missing_points']) or 'None'}",
                    f"Feedback: {analysis['feedback']}",
                ],
            ))
        sections.extend([
            ("Strengths", report_data["strengths"]),
            ("Weak areas", report_data["weak_areas"]),
            ("Final feedback", [report_data["final_feedback"]]),
            ("Ethical note", [report_data["ethical_note"]]),
        ])
    else:
        y = _write_wrapped_line(pdf, f"Question: {report_data['question']}", 42, y)
        y -= 8

        sections = [
            ("Scores", [
                f"Overall: {report_data['final_score']['overall_score']}/100 ({report_data['final_score']['performance_band']})",
                f"Technical: {report_data['technical_analysis']['technical_score']}/100 | Communication: {report_data['audio_analysis']['communication_score']}/100 | Visual: {report_data['visual_analysis']['visual_score']}/100",
            ]),
            ("Key metrics", [
                f"Face presence: {report_data['visual_analysis']['face_presence']}% | Iris-centering proxy: {report_data['visual_analysis']['eye_contact_score']}% | Head stability: {report_data['visual_analysis']['head_stability']}%",
                f"Duration: {report_data['audio_analysis']['duration_seconds']}s | Rate: {report_data['audio_analysis']['words_per_minute']} WPM | Pauses: {report_data['audio_analysis']['pause_count']}",
                f"Semantic similarity: {report_data['technical_analysis']['semantic_similarity']} | Concept coverage: {report_data['technical_analysis']['concept_coverage']}%",
            ]),
            ("Strengths", report_data["feedback"]["strengths"]),
            ("Areas to improve", report_data["feedback"]["weak_areas"]),
            ("Suggestions", report_data["feedback"]["suggestions"]),
            ("Ethical note", [report_data["ethical_note"]]),
        ]
    for heading, lines in sections:
        y = new_page_if_needed(y)
        pdf.setFont("Helvetica-Bold", 12)
        pdf.drawString(42, y, heading)
        y -= 18
        pdf.setFont("Helvetica", 10)
        for line in lines:
            y = new_page_if_needed(y)
            y = _write_wrapped_line(pdf, f"• {line}", 48, y)
        y -= 7
    pdf.save()
    return file_path
