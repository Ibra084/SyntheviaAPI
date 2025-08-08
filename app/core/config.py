# BEFORE
# from pydantic import BaseSettings, EmailStr, AnyHttpUrl

# AFTER
from pydantic_settings import BaseSettings
from pydantic import EmailStr, AnyHttpUrl
from typing import List
from typing import List

class Settings(BaseSettings):
    ENV: str = "development"
    SECRET_KEY: str
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 60
    ALGORITHM: str = "HS256"

    CORS_ORIGINS: List[AnyHttpUrl] = []

    GMAIL_USER: EmailStr
    GMAIL_PASS: str

    GOOGLE_APPLICATION_CREDENTIALS: str

    GOOGLE_API_KEY: str

    class Config:
        env_file = ".env"

settings = Settings()
