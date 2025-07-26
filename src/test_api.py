import base64

import Levenshtein
import numpy as np
import pytest
import requests

from config import config
from noises.gaussian import GaussianNoise
from noises.white import WhiteNoise
from utils.audio import apply_noise, get_allowed_levenshtein_distance, get_wav_info

noise_modules = {
    "gaussian": GaussianNoise(),
    "white": WhiteNoise(),
}


@pytest.mark.run(order=1)
class TestNoNoise:
    results = {}

    @pytest.mark.parametrize("length", config.TEST_LENGTHS)
    def test_encode_decode(self, length):
        original = "".join(np.random.choice(list("0123456789"), length))
        expected = original if original else "0"

        try:
            # Encode
            r = requests.post(
                f"{config.BASE_URL}/encode",
                json={"text": original},
                timeout=config.REQUEST_TIMEOUT,
            )
            assert r.status_code == 200
            b64 = r.json()["data"]
            wav_bytes = base64.b64decode(b64)
            info = get_wav_info(wav_bytes)
            assert info["duration"] <= 10

            # Decode
            r = requests.post(
                f"{config.BASE_URL}/decode",
                json={"data": b64},
                timeout=config.REQUEST_TIMEOUT,
            )
            assert r.status_code == 200
            decoded = r.json()["text"]
            assert decoded == expected

            self.results[length] = {"status": "passed", "lev": 0}
        except Exception as e:
            lev = (
                Levenshtein.distance(expected, decoded) if "decoded" in locals() else -1
            )
            self.results[length] = {"status": "failed", "lev": lev}
            pytest.fail(str(e))


@pytest.mark.run(order=2)
class TestWithNoise:
    results = {key: {} for key in noise_modules}

    @pytest.mark.parametrize("length", config.TEST_LENGTHS)
    @pytest.mark.parametrize("noise_type", noise_modules.keys())
    def test_with_noise(self, noise_type, length):
        noise_module = noise_modules[noise_type]
        for level in noise_module.get_variants():
            original = "".join(np.random.choice(list("0123456789"), length))
            expected = original if original else "0"
            try:
                # Encode
                r = requests.post(
                    f"{config.BASE_URL}/encode",
                    json={"text": original},
                    timeout=config.REQUEST_TIMEOUT,
                )
                assert r.status_code == 200
                wav_bytes = base64.b64decode(r.json()["data"])
                signal = apply_noise(lambda x: noise_module.apply(x, level), wav_bytes)
                b64 = base64.b64encode(signal).decode()

                # Decode
                r = requests.post(
                    f"{config.BASE_URL}/decode",
                    json={"data": b64},
                    timeout=config.REQUEST_TIMEOUT,
                )
                assert r.status_code == 200
                decoded = r.json()["text"]

                lev = Levenshtein.distance(expected, decoded)
                assert lev <= get_allowed_levenshtein_distance(len(expected))
                self.results[noise_type].setdefault(length, {})[level] = {
                    "status": "passed",
                    "lev": lev,
                }
            except Exception as e:
                lev = (
                    Levenshtein.distance(expected, decoded)
                    if "decoded" in locals()
                    else -1
                )
                self.results[noise_type].setdefault(length, {})[level] = {
                    "status": "failed",
                    "lev": lev,
                }
                pytest.fail(
                    f"{noise_type} failed for length={length}, level={level}: {e}"
                )
