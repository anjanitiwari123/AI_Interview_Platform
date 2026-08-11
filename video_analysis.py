
from __future__ import annotations
import math
import cv2
try:
    import mediapipe as mp
except ImportError:  
    mp = None

LEFT_IRIS = [468, 469, 470, 471, 472]
RIGHT_IRIS = [473, 474, 475, 476, 477]
LEFT_EYE_CORNERS = (33, 133)
RIGHT_EYE_CORNERS = (362, 263)

def _mean_point(landmarks, indices, width, height):
    x = sum(landmarks[index].x for index in indices) / len(indices) * width
    y = sum(landmarks[index].y for index in indices) / len(indices) * height
    return x, y

def _distance(point_a, point_b):
    return math.dist(point_a, point_b)

def _eye_centering_score(landmarks, width, height):
    try:
        scores = []
        for iris_indices, corners in ((LEFT_IRIS, LEFT_EYE_CORNERS), (RIGHT_IRIS, RIGHT_EYE_CORNERS)):
            iris = _mean_point(landmarks, iris_indices, width, height)
            corner_a = (landmarks[corners[0]].x * width, landmarks[corners[0]].y * height)
            corner_b = (landmarks[corners[1]].x * width, landmarks[corners[1]].y * height)
            eye_width = max(_distance(corner_a, corner_b), 1.0)
            eye_center = ((corner_a[0] + corner_b[0]) / 2, (corner_a[1] + corner_b[1]) / 2)
            scores.append(max(0.0, 1 - _distance(iris, eye_center) / (0.25 * eye_width)))
        return sum(scores) / len(scores) * 100
    except (IndexError, ZeroDivisionError):
        return None


def _head_motion(nose_points, face_widths):
    if len(nose_points) < 2:
        return 0.0
    normalized_moves = []
    for index in range(1, len(nose_points)):
        movement = _distance(nose_points[index - 1], nose_points[index])
        normalized_moves.append(movement / max(face_widths[index], 1.0))
    average_move = sum(normalized_moves) / len(normalized_moves)
    return round(max(0.0, min(100.0, 100 * (1 - average_move / 0.015))), 1)


def analyze_video(video_path, sample_every_seconds=0.5, max_samples=300):
    if mp is None:
        raise RuntimeError("MediaPipe is not installed. Run: pip install mediapipe")

    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise ValueError("Could not open the uploaded video. Try MP4, MOV, or AVI.")

    fps = capture.get(cv2.CAP_PROP_FPS) or 25.0
    total_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    duration_seconds = round(total_frames / fps, 2) if total_frames else 0.0
    frame_step = max(1, int(fps * sample_every_seconds))

    sampled_frames = 0
    face_frames = 0
    eye_scores = []
    nose_points = []
    face_widths = []
    frame_index = 0

    face_mesh = mp.solutions.face_mesh.FaceMesh(
        static_image_mode=False,
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    )
    try:
        while sampled_frames < max_samples:
            ok, frame = capture.read()
            if not ok:
                break
            if frame_index % frame_step != 0:
                frame_index += 1
                continue

            sampled_frames += 1
            height, width = frame.shape[:2]
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            result = face_mesh.process(rgb_frame)
            if result.multi_face_landmarks:
                face_frames += 1
                landmarks = result.multi_face_landmarks[0].landmark
                eye_score = _eye_centering_score(landmarks, width, height)
                if eye_score is not None:
                    eye_scores.append(eye_score)
                nose_points.append((landmarks[1].x * width, landmarks[1].y * height))
                left_cheek = (landmarks[234].x * width, landmarks[234].y * height)
                right_cheek = (landmarks[454].x * width, landmarks[454].y * height)
                face_widths.append(_distance(left_cheek, right_cheek))
            frame_index += 1
    finally:
        face_mesh.close()
        capture.release()

    if sampled_frames == 0:
        raise ValueError("No readable frames were found in the uploaded video.")

    face_presence = round(face_frames / sampled_frames * 100, 1)
    eye_contact_score = round(sum(eye_scores) / len(eye_scores), 1) if eye_scores else 0.0
    head_stability = _head_motion(nose_points, face_widths)
    visual_score = round(0.45 * face_presence + 0.35 * eye_contact_score + 0.20 * head_stability, 1)

    return {
        "duration_seconds": duration_seconds,
        "sampled_frames": sampled_frames,
        "face_frames": face_frames,
        "face_presence": face_presence,
        "eye_contact_score": eye_contact_score,
        "head_stability": head_stability,
        "visual_score": visual_score,
        "method_note": (
            "Eye score estimates iris centering in the frame; it is a camera-facing proxy, "
            "not a measure of eye contact, confidence, personality, or truthfulness."
        ),
    }
