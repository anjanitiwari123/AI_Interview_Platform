
from __future__ import annotations
import io
import os
import subprocess
from functools import lru_cache
import numpy as np

try:
    import soundfile as sf
except ImportError:
    sf = None


try:
    import librosa
except ImportError:
    librosa = None


def _to_mono_float32(samples, channel_count):
    """Normalise PyAV's planar/packed frame arrays to mono float samples."""
    samples = np.asarray(samples)
    if samples.ndim == 1:
        if channel_count > 1 and samples.size % channel_count == 0:
            mono = samples.reshape(-1, channel_count).mean(axis=1)
        else:
            mono = samples
    elif samples.shape[0] == channel_count:
        mono = samples.mean(axis=0)
    elif samples.shape[-1] == channel_count:
        mono = samples.mean(axis=-1)
    elif samples.shape[0] == 1 and channel_count > 1 and samples.shape[1] % channel_count == 0:
        mono = samples.reshape(-1, channel_count).mean(axis=1)
    else:
        raise ValueError(f"Unsupported microphone frame shape {samples.shape} for {channel_count} channel(s).")

    if np.issubdtype(mono.dtype, np.integer):
        mono = mono.astype(np.float32) / np.iinfo(mono.dtype).max
    else:
        mono = mono.astype(np.float32)
    return np.clip(mono, -1.0, 1.0)


def save_webcam_audio(audio_frames, output_path):
    """Save WebRTC microphone frames as a clean, mono WAV for Whisper.

    WebRTC commonly supplies planar 48 kHz signed-integer frames.  Converting
    explicitly avoids channel/interleaving mistakes that make speech appear
    nearly silent to the recogniser.
    """
    if not audio_frames:
        raise ValueError("No microphone audio was recorded. Check microphone permission and try again.")
    if sf is None:
        raise RuntimeError("soundfile is not installed. Run: pip install soundfile")

    sample_rate = audio_frames[0][1]
    chunks = []
    for frame_data in audio_frames:
        samples, frame_rate = frame_data[:2]
        channel_count = frame_data[2] if len(frame_data) > 2 else 1
        if frame_rate != sample_rate:
            raise ValueError("Microphone sample rate changed during recording. Please record the answer again.")
        chunks.append(_to_mono_float32(samples, channel_count))

    audio = np.concatenate(chunks, axis=0)
    return _write_normalized_wav(audio, sample_rate, output_path)


def _write_normalized_wav(audio, sample_rate, output_path):
    """Validate, normalise, and write a mono 16 kHz WAV for Whisper."""
    audio = np.asarray(audio, dtype=np.float32).reshape(-1)
    duration = len(audio) / sample_rate
    peak = float(np.max(np.abs(audio))) if len(audio) else 0.0
    rms = float(np.sqrt(np.mean(np.square(audio)))) if len(audio) else 0.0
    if duration < 0.35 or peak < 0.003 or rms < 0.0008:
        raise ValueError(
            "Very little microphone speech was captured. Ensure the browser is using the correct "
            "microphone, raise its input level, then record again."
        )

    target_rate = 16000
    if sample_rate != target_rate:
        if librosa is None:
            raise RuntimeError("librosa is required to resample browser audio. Run: pip install librosa")
        audio = librosa.resample(audio, orig_sr=sample_rate, target_sr=target_rate)
        sample_rate = target_rate
    sf.write(str(output_path), audio, sample_rate, subtype="PCM_16")
    return output_path


def save_recorded_audio(audio_bytes, output_path):
    """Save the WAV produced by Streamlit's browser-native audio recorder.

    Unlike a live WebRTC frame callback, ``st.audio_input`` supplies a complete
    browser-recorded WAV. This avoids packet-layout and timing errors that can
    make Whisper hear unrelated speech.
    """
    if sf is None:
        raise RuntimeError("soundfile is not installed. Run: pip install soundfile")
    samples, sample_rate = sf.read(io.BytesIO(audio_bytes), dtype="float32", always_2d=True)
    if samples.size == 0:
        raise ValueError("The recording is empty. Record your answer again.")
    return _write_normalized_wav(samples.mean(axis=1), sample_rate, output_path)

def extract_audio_from_video(video_path, output_path):
    command = [
        "ffmpeg",
        "-y",
        "-i",
        str(video_path),
        "-vn",
        "-ac",
        "1",
        "-ar",
        "16000",
        "-loglevel",
        "error",
        str(output_path),
    ]
    try:

        subprocess.run(
            command,
            check=True,
            capture_output=True,
            text=True
        )

    except FileNotFoundError as error:
        raise RuntimeError(
            "ffmpeg not found. Install ffmpeg first."
        ) from error


    except subprocess.CalledProcessError as error:
        raise RuntimeError(
            f"Audio extraction failed: {error.stderr}"
        ) from error



    if not os.path.exists(output_path):
        raise RuntimeError(
            "Audio file was not created."
        )


    return output_path





@lru_cache(maxsize=3)
def _load_whisper_model(model_name):
    """Keep the selected Whisper model in memory between Streamlit reruns."""
    try:
        import whisper
    except ImportError as error:
        raise RuntimeError("Install whisper using pip install openai-whisper") from error
    return whisper.load_model(model_name)


def transcribe_audio(audio_path, model_name="base"):
    """
    Convert speech into text using Whisper.
    """
    try:
        model = _load_whisper_model(model_name)
        result = model.transcribe(
            str(audio_path),
            fp16=False,
            verbose=False
        )
        text = result.get(
            "text",
            ""
        ).strip()
        return text
    except Exception as error:
        raise RuntimeError(
            f"Whisper failed: {error}"
        ) from error
def _count_pauses(
        rms_db,
        hop_seconds,
        minimum_pause_seconds=0.45
):
    """
    Estimate silence gaps.
    """
    if len(rms_db)==0:
        return 0
    threshold = max(
        float(np.percentile(rms_db,20)),
        float(np.max(rms_db)-35)
    )
    silent = rms_db < threshold
    min_frames = max(
        1,
        int(minimum_pause_seconds / hop_seconds)
    )
    count = 0
    running = 0
    for value in silent:
        if value:
            running += 1
        else:
            running = 0
        if running == min_frames:
            count += 1
    return count

def analyze_audio(audio_path, transcript):

    """
    Calculate speech communication metrics.
    """
    if librosa is None:
        raise RuntimeError(
            "Install librosa using pip install librosa"
        )
    samples, sample_rate = librosa.load(
        str(audio_path),
        sr=16000,
        mono=True
    )
    duration = len(samples)/sample_rate
    if duration <= 0:
        raise ValueError(
            "Empty audio file"
        )
    rms = librosa.feature.rms(
        y=samples
    )[0]


    rms_db = librosa.amplitude_to_db(
        np.maximum(rms,1e-10),
        ref=np.max
    )
    hop_seconds = 512/sample_rate
    pause_count = _count_pauses(
        rms_db,
        hop_seconds
    )
    words = [
        word
        for word in transcript.split()
        if any(c.isalnum() for c in word)
    ]
    word_count = len(words)
    words_per_minute = round(
        word_count/duration*60,
        1
    )
    spectral_flatness = float(
        np.mean(
            librosa.feature.spectral_flatness(
                y=samples
            )
        )
    )
    voiced_energy_ratio = float(
        np.mean(
            rms > np.percentile(rms,20)
        )
    )
    loudness_consistency = (
        1 -
        min(
            1.0,
            float(np.std(rms_db))/25
        )
    )
    clarity_proxy = round(
        100 *
        (
            0.45*(1-spectral_flatness)
            +
            0.30*voiced_energy_ratio
            +
            0.25*loudness_consistency
        ),
        1
    )
    rate_score = max(
        0,
        100-abs(words_per_minute-145)*1.3
    )
    pause_score = max(
        0,
        100-max(
            0,
            pause_count-max(2,duration/25)
        )*7
    )
    communication_score = round(
        0.45*rate_score
        +
        0.30*pause_score
        +
        0.25*clarity_proxy,
        1
    )
    return {
        "duration_seconds":
            round(duration,1),
        "word_count":
            word_count,
        "words_per_minute":
            words_per_minute,
        "pause_count":
            pause_count,
        "clarity_proxy":
            clarity_proxy,
        "communication_score":
            communication_score,
        "transcript":
            transcript,
        "method_note":
            (
            "Communication score is calculated using "
            "speech rate, pause estimation and acoustic features. "
            "It does not judge accent or personality."
            )
    }
