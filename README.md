# Multimodal AI Interview Evaluation System

An explainable Streamlit project for practising technical interview answers from a recorded video. It combines observable video signals, speech delivery metrics, and reference-based NLP evaluation into a simple report.

> This is an educational practice tool, **not** an automated hiring or ranking system. It intentionally does not infer personality, emotion, confidence, honesty, or employability.

## What it does

1. Samples frames from an uploaded interview video and uses MediaPipe Face Mesh to estimate face visibility, iris-centering and head-motion stability.
2. Extracts audio with `ffmpeg`, transcribes it with a local pretrained Whisper model, then calculates speaking rate, silence runs and acoustic clarity proxies using Librosa.
3. Uses `all-MiniLM-L6-v2` Sentence Transformer embeddings to compare the candidate answer against a supplied reference answer. It also checks an explicit concept rubric.
4. Combines the three module scores with a transparent formula and generates feedback, JSON, and PDF reports.

## Project structure

```text
multimodal_interview_ai/
├── app.py                 # Streamlit UI and pipeline orchestration
├── video_analysis.py      # OpenCV + MediaPipe visual analysis
├── audio_analysis.py      # ffmpeg, Whisper and Librosa analysis
├── nlp_analysis.py        # Sentence embeddings and concept coverage
├── scoring.py             # Explainable weighted fusion
├── feedback.py            # Traceable rule-based feedback
├── report.py              # JSON/PDF report writers
├── requirements.txt
├── README.md
└── reports/               # Created automatically; no database used
```

The code uses only small, focused functions. There are no custom classes, database, FastAPI server, Docker files, or opaque neural fusion model.

## Installation

### 1. Create and activate a virtual environment

```bash
cd multimodal_interview_ai
python -m venv .venv
source .venv/bin/activate          # macOS / Linux
# .venv\Scripts\activate           # Windows PowerShell
python -m pip install --upgrade pip
```

### 2. Install Python packages

```bash
pip install -r requirements.txt
```

Whisper downloads the selected model the first time it runs. Start with `tiny` for a quick smoke test and use `base` for a better demo.

### 3. Install ffmpeg

Whisper processes audio, so the app first converts the uploaded video's audio track to a 16 kHz WAV file.

```bash
# macOS (Homebrew)
brew install ffmpeg

# Ubuntu/Debian
sudo apt update && sudo apt install ffmpeg
```

On Windows, install ffmpeg from its official builds and ensure its `bin` directory is in `PATH`.

### 4. Run the app

```bash
streamlit run app.py
```

Open the local URL printed by Streamlit, upload a short MP4/MOV video, enter the question, a reference answer, and (ideally) a comma-separated rubric. The candidate answer defaults to the Whisper transcript; use the optional override only to correct a transcription issue.

## Demo inputs

**Question**

```text
Explain CNN architecture.
```

**Reference answer**

```text
A convolutional neural network uses convolution filters to learn local image patterns.
Each convolution is usually followed by a non-linear activation. Pooling reduces spatial
size, and later feature maps are passed to a classifier head for prediction. Training
learns the filter weights with backpropagation.
```

**Expected concepts**

```text
convolution, filters, activation, pooling, classifier, backpropagation
```

## Example output

The precise score varies with video quality and model versions. A representative JSON report looks like this:

```json
{
  "visual_analysis": {
    "face_presence": 95.0,
    "eye_contact_score": 82.3,
    "head_stability": 78.1,
    "visual_score": 84.3
  },
  "audio_analysis": {
    "duration_seconds": 120.0,
    "words_per_minute": 145.0,
    "pause_count": 4,
    "communication_score": 80.0
  },
  "technical_analysis": {
    "semantic_similarity": 0.86,
    "concept_coverage": 75.0,
    "technical_score": 82.7
  },
  "final_score": {
    "overall_score": 81.2,
    "formula": "0.50 × technical + 0.30 × communication + 0.20 × visual"
  }
}
```

Each run writes timestamped JSON and PDF files to `reports/`. The Streamlit page also presents download buttons.

## How the modules work

### 1. `video_analysis.py` — visual engagement signals

**Purpose:** convert a video into lightweight, observable camera-facing signals.

**Input → output:** a video path → duration, sampled frame count, face-presence percentage, iris-centering proxy, normalized head stability and `visual_score`.

**Important code ideas**

- `cv2.VideoCapture(video_path)` reads the video frame-by-frame. `frame_step` samples every 0.5 seconds instead of running inference on every frame, which lowers compute while retaining coverage.
- `FaceMesh(..., refine_landmarks=True)` returns 3D-normalized facial landmark coordinates. The app uses iris landmarks (468–477), eye corners, the nose point and cheek points.
- Iris distance from each eye's center is normalized by eye width. This makes the proxy less sensitive to resolution and face size.
- Head movement is computed as consecutive nose-point motion divided by detected face width. It is deliberately a motion signal, not a confidence claim.
- `visual_score = 0.45 × face_presence + 0.35 × iris_centering + 0.20 × head_stability`.

**AI concepts:** landmark detection, normalized geometric features, temporal frame sampling, pretrained computer-vision inference.

**Limits:** an iris-centering score is only a rough camera-facing proxy. Glasses, profile views, webcam position, lighting, accessibility needs, and video quality affect it. Never call it personality, eye contact certainty, confidence, or lie detection.

### 2. `audio_analysis.py` — speech communication signals

**Purpose:** transcribe the answer and calculate understandable delivery metrics.

**Input → output:** video path/audio path plus transcript → transcript, duration, word count, WPM, pause count, acoustic clarity proxy and `communication_score`.

**Important code ideas**

- `ffmpeg -vn -ac 1 -ar 16000` removes video, converts to mono, and resamples to 16 kHz WAV.
- `whisper.load_model(model_name)` loads a pretrained local encoder-decoder model; `model.transcribe(...)` returns text.
- `librosa.feature.rms` produces short-time energy. Long runs below a relative loudness threshold count as estimated pauses.
- Spectral flatness, proportion of voiced energy and RMS consistency form a documented **clarity proxy**. They do not assess accent, grammar, or pronunciation accuracy.
- A target pace of 120–170 WPM is a project heuristic. The score combines rate (45%), pause behavior (30%) and acoustic proxy (25%).

**AI concepts:** automatic speech recognition, time-frequency audio features, RMS energy, spectral flatness, signal-level feature engineering.

**Whisper explanation:** Whisper converts audio into log-Mel spectrogram features. Its encoder produces audio representations and an autoregressive decoder predicts text tokens conditioned on those representations. It is pretrained on large-scale multilingual speech data. This project uses it only for transcription; the acoustic metrics come separately from Librosa.

### 3. `nlp_analysis.py` — technical answer understanding

**Purpose:** evaluate answer relevance against an explicit, human-written technical reference answer and rubric.

**Input → output:** question, candidate answer, reference answer, optional concept list → cosine similarity, covered/missing concepts, coverage percentage and `technical_score`.

**Important code ideas**

- `SentenceTransformer("all-MiniLM-L6-v2")` encodes sentences into dense vectors. It is lightweight enough for a portfolio demo but still captures semantic similarity better than raw keyword matching.
- Normalized embedding vectors are compared with cosine similarity. A score closer to 1 means the vectors point in similar directions.
- An explicit comma-separated rubric makes the missing-concept explanation auditable. If no rubric is supplied, the project transparently derives simple candidate terms from the reference answer as a fallback.
- `technical_score = 100 × (0.70 × semantic_similarity + 0.30 × concept_coverage)`.

**AI concepts:** Transformer contextual embeddings, sentence transformers, cosine similarity, human-in-the-loop rubric evaluation.

**Why a reference answer is required:** similarity needs a meaningful target. Comparing an answer only with a question could reward topical wording without verifying correctness. A reference answer keeps the evaluation scoped and explainable.

### 4. `scoring.py` — multimodal score fusion

**Purpose:** combine module scores without claiming a complex end-to-end neural model.

**Input → output:** technical, communication and visual scores → overall score, performance band, weights and formula.

```text
overall = 0.50 × technical + 0.30 × communication + 0.20 × visual
```

Technical quality receives the most weight because this is a technical-interview practice project. Communication influences how well an answer is delivered. Visual signals receive the lowest weight because they are more sensitive to the physical recording setup. Fixed weighted fusion is chosen because every score contribution is inspectable, easy to tune using user feedback, and does not need a labelled multimodal hiring dataset.

### 5. `feedback.py` — feedback generation

**Purpose:** turn metric thresholds into practical, traceable suggestions.

**Input → output:** all metric dictionaries → strengths, weak areas, suggestions, and a summary.

Every generated statement maps to a visible condition—for example, missing rubric concepts produce a suggested answer structure. A rules-based approach is preferred here over generated feedback so it remains deterministic and easy to explain in an interview.

### 6. `report.py` — portable results

**Purpose:** persist one analysis result with no database.

**Input → output:** full result dictionary → timestamped JSON and PDF.

JSON preserves all raw metrics for inspection. The PDF is a concise human-readable summary made with ReportLab. Timestamped filenames avoid silently overwriting a previous run.

### 7. `app.py` — Streamlit UI

**Purpose:** collect inputs, call functions in sequence, display metrics, and offer report downloads.

**Input → output:** upload and text fields → dashboard tabs plus two downloadable files.

The UI uses `tempfile.TemporaryDirectory()` for uploaded media, so it is removed after processing. Only generated reports are retained locally in `reports/`.

## Interview questions and concise answers

**Why sample video frames instead of analysing every frame?**  Face Mesh inference on every frame is unnecessarily expensive for an interview-practice dashboard. Time-based sampling preserves broad coverage and makes latency reasonable. I would validate the sampling interval against a small manually reviewed set.

**How does Face Mesh help estimate gaze?**  It returns normalized facial landmark coordinates. I compare an iris center with the eye center and normalize by eye width. That indicates whether the visible eyes are roughly centered, but it is only a camera-facing proxy—not an eye-contact or confidence detector.

**Why normalize head movement by face width?**  Raw pixel movement changes with resolution and distance from camera. Face-width normalization makes the feature more comparable across videos.

**How does Whisper work at a high level?**  Whisper transforms audio into log-Mel spectrograms. A Transformer encoder processes the acoustic representation and a decoder generates text tokens. I use its output as the candidate answer transcript.

**How are pauses estimated?**  I compute short-time RMS energy with Librosa, mark continuous relative-low-energy regions, and count runs longer than 0.45 seconds. It is an estimate and can be affected by background noise or recording levels.

**Why use sentence embeddings instead of keyword matching alone?**  Keywords miss paraphrases. Sentence Transformer embeddings capture semantic context, while the separate explicit rubric retains a transparent coverage check. Combining them balances flexibility and explainability.

**What is cosine similarity?**  It measures the angle between two embedding vectors: `a · b / (||a|| ||b||)`. With L2-normalized vectors it reduces to their dot product. Higher values indicate semantic closeness, not factual proof.

**Why not train a multimodal neural fusion model?**  A reliable trained fusion model needs a sizeable, representative, ethically collected labelled dataset and careful bias evaluation. For a student project, fixed weighted fusion is more defensible, inspectable, and appropriate.

**How would you evaluate this project?**  I would create consented practice-video cases with human rubric scores, measure correlation and ranking consistency for each module, inspect errors across lighting/accent/camera conditions, and tune thresholds only on a validation split. I would never validate it as an autonomous hiring decision system.

## Good talking points for the project demo

- Start with the explicit limitation: it supports mock-interview practice, not hiring automation.
- Show one sample video, the automatically generated transcript, the reference answer/rubric, then explain every score contribution.
- Point to the raw JSON report to demonstrate reproducibility and traceability.
- Explain trade-offs: `base` Whisper for speed/quality, MiniLM for local lightweight embeddings, and frame sampling for responsive UX.
- Suggest next steps only after describing current limitations: consent-based evaluation data, multilingual ASR testing, manually curated question rubrics, and threshold calibration.

## Resume description

**Multimodal AI Interview Evaluation System | Python, Streamlit, OpenCV, MediaPipe, Whisper, Librosa, Sentence Transformers**  
Built an explainable mock-interview evaluation app that samples video frames for facial landmark-based engagement proxies, transcribes responses with Whisper, extracts speech-rate/pause audio features, and evaluates answers with Sentence Transformer cosine similarity plus a concept rubric. Designed transparent weighted score fusion and generated downloadable JSON/PDF feedback reports; documented limitations to avoid personality, confidence, or hiring predictions.

## Responsible-use note

The project should be presented as a personal practice/coaching system. Do not use it to screen, rank, reject, or make decisions about real candidates. Any future extension should include informed consent, data minimisation, accessibility review, human oversight, calibration by subgroup, and a clear way to challenge or opt out of feedback.
