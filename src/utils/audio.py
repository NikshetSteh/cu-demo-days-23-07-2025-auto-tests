import io
import wave
from typing import Any, Dict

import numpy as np

from config import config


def get_allowed_levenshtein_distance(text_length: int) -> int:
    return config.LEVENSHTEIN_ABSOLUTE_TOLERANCE + int(
        text_length * config.LEVENSHTEIN_RELATIVE_TOLERANCE
    )


def get_wav_info(wav_bytes: bytes) -> Dict[str, Any]:
    with io.BytesIO(wav_bytes) as buffer:
        with wave.open(buffer, "rb") as wf:
            return {
                "nchannels": wf.getnchannels(),
                "sampwidth": wf.getsampwidth(),
                "framerate": wf.getframerate(),
                "nframes": wf.getnframes(),
                "duration": wf.getnframes() / float(wf.getframerate()),
            }


def apply_noise(noise_fn, wav_bytes: bytes) -> bytes:
    with wave.open(io.BytesIO(wav_bytes), "r") as wf:
        signal = np.frombuffer(wf.readframes(wf.getnframes()), dtype=np.int16)

    noisy_signal = noise_fn(signal)

    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as wf:
        wf.setnchannels(config.EXPECTED_CHANNELS)
        wf.setsampwidth(config.EXPECTED_SAMPWIDTH)
        wf.setframerate(config.EXPECTED_FRAMERATE)
        wf.writeframes(noisy_signal.tobytes())
    return buffer.getvalue()
