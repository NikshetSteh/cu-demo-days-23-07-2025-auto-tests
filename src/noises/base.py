from abc import ABC, abstractmethod

import numpy as np


class BaseNoise(ABC):
    @abstractmethod
    def get_variants(self):
        pass

    @abstractmethod
    def apply(self, audio: np.ndarray, level) -> np.ndarray:
        pass
