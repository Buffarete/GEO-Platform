from pydantic import BaseSettings

class Settings(BaseSettings):
    database_url: str = "postgresql://user:password@db:5432/geo"
    secret_key: str = "YOUR_SECRET_KEY"
    algorithm: str = "HS256"

settings = Settings()
