import numpy as np

from .base import BaseNoise


class GaussianNoise(BaseNoise):
    def __init__(self, sigmas=[0.01, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35]):
        self.sigmas = sigmas

    def get_variants(self):
        return self.sigmas

    def apply(self, audio: np.ndarray, sigma) -> np.ndarray:
        audio = audio.astype(np.float32) / np.iinfo(audio.dtype).max
        noise = np.random.normal(0, sigma, audio.shape).astype(np.float32)
        noisy = np.clip(audio + noise, -1.0, 1.0)
        return (noisy * np.iinfo(np.int16).max).astype(np.int16)
