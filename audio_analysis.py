
from __future__ import annotations
import os
import subprocess
import numpy as np


try:
    import librosa
except ImportError:
    librosa = None

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





def transcribe_audio(audio_path, model_name="base"):
    """
    Convert speech into text using Whisper.
    """

    try:

        import whisper

    except ImportError as error:

        raise RuntimeError(
            "Install whisper using pip install openai-whisper"
        ) from error



    try:

        model = whisper.load_model(
            model_name
        )


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