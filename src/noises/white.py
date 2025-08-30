import numpy as np

from .base import BaseNoise


class WhiteNoise(BaseNoise):
    def __init__(self, snrs=[30, 20, 10, 5]):
        self.snrs = snrs

    def get_variants(self):
        return self.snrs

    def apply(self, audio: np.ndarray, snr_db) -> np.ndarray:
        audio = audio.astype(np.float32) / np.iinfo(audio.dtype).max
        power_signal = np.mean(audio**2)
        if power_signal == 0:
            return (audio * np.iinfo(np.int16).max).astype(np.int16)
        power_noise = power_signal / (10 ** (snr_db / 10.0))
        noise = np.random.normal(0, np.sqrt(power_noise), audio.shape).astype(
            np.float32
        )
        noisy = np.clip(audio + noise, -1.0, 1.0)
        return (noisy * np.iinfo(np.int16).max).astype(np.int16)
