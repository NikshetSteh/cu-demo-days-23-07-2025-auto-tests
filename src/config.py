from pydantic_settings import BaseSettings


class Config(BaseSettings):
    BASE_URL: str = "http://localhost:8000"
    REQUEST_TIMEOUT: int = 60

    EXPECTED_CHANNELS: int = 1
    EXPECTED_SAMPWIDTH: int = 2
    EXPECTED_FRAMERATE: int = 44100
    MAX_DURATION_S: int = 10

    TEST_LENGTHS: list[int] = [1, 5, 10, 100, 1000, 10_000, 20_000, 30_000]
    LEVENSHTEIN_ABSOLUTE_TOLERANCE: int = 1
    LEVENSHTEIN_RELATIVE_TOLERANCE: float = 0.05

    NOISE_TEST_SEQUENCE: str = "12345"


config = Config()
