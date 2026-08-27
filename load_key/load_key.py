import os
from dotenv import load_dotenv

load_dotenv()


class Settings:
    OPENAI_API_KEY_NAME: str = os.getenv("OPEN_API_KEY_NAME", "")
    OPENAI_API_KEY: str = os.getenv("OPENAI_API_KEY", "")
    GEMINI_API_KEY: str = os.getenv("GEMINI_API_KEY", "")
    GEMINI_API_KEY_NAME: str = os.getenv("GEMINI_API_KEY_NAME", "")

    @classmethod
    def validate(cls):
        if not cls.OPENAI_API_KEY:
            raise RuntimeError(
                "Missing OPENAI_API_KEY. Please set it in your .env file."
            )



