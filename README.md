# Real-Time AI Interview Practice Platform

A beginner-friendly Streamlit mock-interview app. The candidate opens a browser camera and microphone, answers live questions, and receives technical practice feedback. It is an educational coaching tool, not an automated hiring or ranking system.


## 🚀 Live Demo

🔗 **Streamlit App:**  
https://aiinterviewplatform-at22.streamlit.app/

---

## Live interview flow

```text
Start Interview
    ↓
Upload a PDF/DOCX/TXT resume
    ↓
Ollama generates a question from the resume's skills and projects
    ↓
Camera preview opens in the browser
    ↓
Candidate records an answer with the browser-native microphone recorder
    ↓
Whisper transcribes the microphone audio
    ↓
Candidate reviews/corrects the transcript
    ↓
Sentence Transformer + Ollama judge evaluate the answer
    ↓
Next question or final JSON/PDF report
```

The candidate no longer uploads a recorded video and does not type an expected answer. Uploading a resume is now the recommended start: its listed technical skills, projects, and experience are sent to Ollama as the primary question source. The model is instructed not to invent unlisted technologies or projects. A target role is optional and only helps focus the question selection.

## Files and responsibilities

| File | Responsibility |
| --- | --- |
| `app.py` | Streamlit screens, interview state, WebRTC camera preview, browser-native microphone recording, question progression, and downloads. |
| `audio_analysis.py` | Validates browser-recorded WAV audio, Whisper transcription, and optional delivery metrics. |
| `nlp_analysis.py` | Ollama question generation, Sentence Transformer similarity/semantic coverage, and Ollama rubric judging. |
| `feedback.py` | Builds strengths, weak areas, and final feedback from all questions. |
| `report.py` | Writes interview-level JSON and PDF reports. |
| `video_analysis.py` | Legacy uploaded-video visual analysis; it is retained but not part of the live interview score. |
| `scoring.py` | Legacy multimodal weighted score fusion; retained for the old pipeline but not used by the live technical report. |

## Technical scoring

For every question, Ollama returns:

```json
{
  "question": "Explain CNN architecture.",
  "evaluation_points": [
    "convolution layers extract local features",
    "activation functions add non-linearity",
    "pooling reduces spatial dimensions",
    "classification head produces predictions"
  ],
  "difficulty": "intermediate"
}
```

The final question score is `0–10` and combines:

- Ollama rubric judge: 50%
- Sentence Transformer semantic similarity: 30%
- semantic coverage of the generated evaluation points: 20%

This judges meaning and paraphrases rather than requiring exact expected-answer wording. Each result includes covered points, missing points, feedback, transcript, and score.

## Setup

1. Create and activate a virtual environment.

```bash
python -m venv .venv
source .venv/bin/activate
```

2. Install Python requirements.

```bash
pip install -r requirements.txt
```

3. Install ffmpeg. Whisper uses it to read the recorded WAV file.

```bash
# macOS
brew install ffmpeg
```

4. Install [Ollama](https://ollama.com), start it, and download the model selected in the Streamlit sidebar.

```bash
ollama serve
ollama pull llama3.2
```

5. Run the Streamlit app.

```bash
streamlit run app.py
```

6. In the browser:

   1. Upload a PDF, DOCX, or TXT resume. Optionally enter a target role or topic, then click **Start Interview**.
   2. Click **START** in the camera panel and allow camera access.
   3. Use **Answer recording** to record your spoken answer, then click **Transcribe and review answer**.
   4. Review the Whisper transcript and correct any words it misheard before clicking **Evaluate reviewed answer**. Only this reviewed transcript is scored.
   5. Repeat until the final report appears, then download JSON or PDF.

## Notes and limits

- First Whisper and Sentence Transformer runs download their model files and can be slower.
- The app needs a working microphone, browser permission, and local Ollama service.
- Scanned/image-only PDFs do not contain extractable text; upload a text-based PDF, DOCX, or TXT resume instead.
- The full resume is used only in the active browser session to generate questions. The downloaded report stores its filename, not its full text.
- `streamlit-webrtc` needs `localhost` or HTTPS to access the camera. A deployed application may also need STUN/TURN configuration.
- The answer recorder uses Streamlit's browser-native WAV capture rather than live WebRTC audio frames. This prevents WebRTC packet/channel layout issues from producing a garbled Whisper transcript. If it reports very little captured speech, verify the browser's selected microphone and its input level, then record again.
- The camera preview is for the live interview experience only. The final score is technical-answer feedback; it does not infer confidence, emotion, personality, or employability.
