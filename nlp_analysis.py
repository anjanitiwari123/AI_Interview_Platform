from __future__ import annotations

import json
import re
import numpy as np
import streamlit as st
from groq import Groq

def _read_json(text):
    text = (
        text
        .replace("```json", "")
        .replace("```", "")
        .strip()
    )
    start = text.find("{")
    end = text.rfind("}") + 1
    if start == -1:
        raise ValueError("Invalid JSON from LLM")
    return json.loads(text[start:end])
def ask_llm(messages):

    client = Groq(
        api_key=st.secrets["GROQ_API_KEY"]
    )

    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=messages,
        temperature=0.2,
        response_format={
            "type":"json_object"
        }
    )

    return _read_json(
        response.choices[0].message.content
    )

def generate_interview_question(
        topic,
        previous_questions,
        resume_text=""
):

    previous = "\n".join(
        previous_questions
    )
    prompt = f"""
You are a technical interviewer.
Generate ONE interview question.
Candidate resume:
{resume_text[:8000]}
Topic:
{topic}
Previous questions:
{previous}
Rules:
- Question must be based on candidate skills/projects.
- Do not invent technologies.
- Create technical evaluation points.
Return JSON:
{{
"question":"",
"evaluation_points":[
"technical concept",
"implementation detail",
"library/algorithm",
"project usage"
],
"difficulty":"easy/intermediate/hard"
}}
Evaluation points must be specific.
Example:
Bad:
"concept understanding"
Good:
"BERT contextual embeddings"
"Transformer fine tuning using HuggingFace"
"Resume classification pipeline"
"""
    result = ask_llm(
        [
            {
                "role":"user",
                "content":prompt
            }
        ]
    )
    return {
        "question":
        result.get("question",""),

        "evaluation_points":
        result.get(
            "evaluation_points",
            []
        ),

        "difficulty":
        result.get(
            "difficulty",
            "intermediate"
        )

    }

def load_embedding_model(model_name):
    import streamlit as st
    from sentence_transformers import SentenceTransformer
    @st.cache_resource
    def load(name):
        return SentenceTransformer(name)
    return load(model_name)

def cosine_similarity(a,b):
    return float(
        np.dot(a,b)
        /
        (
            np.linalg.norm(a)
            *
            np.linalg.norm(b)
        )
    )

def check_rubric(
        answer,
        points,
        model_name
):
    model = load_embedding_model(
        model_name
    )
    sentences = [part.strip() for part in re.split(r"(?<=[.!?])\s+|\n+", answer) if part.strip()]
    answer_units = [answer, *sentences]
    embeddings = model.encode([*answer_units, *points], normalize_embeddings=True)
    covered=[]
    missing=[]
    scores=[]
    point_start = len(answer_units)
    for i,point in enumerate(points):
        score = max(
        cosine_similarity(
        answer_vector,
        embeddings[point_start+i]
        )
        for answer_vector in embeddings[:point_start]
        )
        scores.append(
            round(score,3)
        )
        if score >= 0.30:
            covered.append(point)

        else:
            missing.append(point)
    return (
        scores,
        covered,
        missing
    )
def evaluate_live_answer(
        question_data,
        candidate_answer,
        embedding_model="all-MiniLM-L6-v2"
):
    question = question_data["question"]
    points = [point for point in question_data.get("evaluation_points", []) if str(point).strip()]
    if not points:
        raise ValueError("This question has no evaluation points. Generate another question before recording.")
    if not candidate_answer.strip():

        raise ValueError(
            "Empty answer"
        )
    model = load_embedding_model(
        embedding_model
    )
    vectors = model.encode(
        [
            candidate_answer,
            " ".join(points)
        ],
        normalize_embeddings=True
    )


    semantic = cosine_similarity(
        vectors[0],
        vectors[1]
    )
    semantic_score = max(0.0, min(10.0, (semantic - 0.10) / 0.45 * 10))
    similarities,covered,missing = check_rubric(
        candidate_answer,
        points,
        embedding_model
    )
    coverage = (

        len(covered)
        /
        len(points)

    )*10
    judge = ask_llm(

        [
            {
                "role":"system",
                "content":
                """
You are a senior technical interviewer.

Evaluate answer based on:

1. Technical correctness
2. Implementation knowledge
3. Tools/libraries
4. Project explanation


Give partial marks.

Return JSON:

{
"score":0,
"feedback":"",
"covered_point_indexes":[]
}

Score 0-10.

Score the meaning of the candidate's answer, not exact keyword overlap. A
technically correct paraphrase or a relevant implementation example deserves
credit. Only use a low score when the answer is substantially incorrect,
irrelevant, or missing most expected points. `covered_point_indexes` must use
zero-based indexes from Expected points.
"""
            },

            {
                "role":"user",

                "content":
                f"""
Question:
{question}
Expected points:
{points}
Candidate answer:
{candidate_answer}

"""
            }

        ]
    )
    try:
        judge_score = float(judge.get("score", 0))
    except (TypeError, ValueError):
        judge_score = 0.0
    judge_score = max(0.0, min(10.0, judge_score))
    judge_indexes = judge.get("covered_point_indexes", [])
    if isinstance(judge_indexes, list):
        for index in judge_indexes:
            if isinstance(index, int) and 0 <= index < len(points):
                point = points[index]
                if point not in covered:
                    covered.append(point)
                if point in missing:
                    missing.remove(point)
    coverage = len(covered) / len(points) * 10
    final_score=round(
        (
            judge_score * 0.50
            + coverage * 0.40
            + semantic_score * 0.10

        ),

        1

    )
    return {
        "question":
        question,
        "difficulty": question_data.get("difficulty", "intermediate"),
        "technical_score":
        final_score,
        "llm_score":
        judge_score,
        "semantic_similarity":
        round(
            semantic,
            3
        ),

        "concept_coverage": round(coverage * 10, 1),
        "covered_points":
        covered,
        "missing_points":
        missing,
        "feedback":
        judge.get(
            "feedback",
            ""
        ),

        "method":
        """
Final score:
50% LLM technical evaluation
40% rubric coverage (sentence-level semantic matching plus judge-confirmed points)
10% semantic similarity
"""
    }

def evaluate_reference_answer(
        question,
        candidate_answer,
        reference_answer,
        model_name="all-MiniLM-L6-v2"

):
    model = load_embedding_model(
        model_name
    )
    vectors=model.encode(

        [
            candidate_answer,
            reference_answer
        ],

        normalize_embeddings=True

    )
    similarity=cosine_similarity(
        vectors[0],
        vectors[1]
    )
    score=round(
        similarity*100,
        1
    )
    return {
        "question":
        question,
        "semantic_similarity":
        round(
            similarity,
            3
        ),
        "technical_score":
        score,

        "feedback":
        "Score based on semantic similarity with reference answer."

    }
