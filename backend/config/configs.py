import os
import enum
from typing import Annotated, Any

from pydantic import (
    AnyUrl,
    BeforeValidator,
    HttpUrl,
    MySQLDsn,
    SecretStr,
    computed_field,
    model_validator,
)

from pydantic_core import MultiHostUrl
from pydantic_settings import BaseSettings, SettingsConfigDict
from typing_extensions import Self


def parse_cors(v: Any) -> list[str] | str:
    if isinstance(v, str) and not v.startswith("["):
        return [i.strip() for i in v.split(",")]
    elif isinstance(v, list | str):
        return v
    raise ValueError(v)


class Environment(str, enum.Enum):
    LOCAL = "local"
    STAGING = "staging"
    PRODUCTION = "production"


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env", env_ignore_empty=True, extra="ignore"
    )
    API_V1_STR: str = "/api/v1"
    SECRET_KEY: str
    DOMAIN: str = "localhost"
    ENVIRONMENT: Environment = Environment.LOCAL

    SESSION_COOKIE_NAME: str = "session"
    # 90 days
    SESSION_COOKIE_MAX_AGE: int = 3600 * 24 * 90
    SESSION_COOKIE_SECURE: bool = False

    BROWSER_ID_COOKIE_NAME: str = "bid"
    BROWSER_ID_COOKIE_MAX_AGE: int = 3600 * 24 * 365 * 2

    @computed_field  # type: ignore[misc]
    @property
    def server_host(self) -> str:
        # Use HTTPS for anything other than local development
        if self.ENVIRONMENT == Environment.LOCAL:
            return f"http://{self.DOMAIN}"
        return f"https://{self.DOMAIN}"

    BACKEND_CORS_ORIGINS: Annotated[
        list[AnyUrl] | str, BeforeValidator(parse_cors)
    ] = []

    PROJECT_NAME: str = "Leg.ai"


    LOCAL_FILE_STORAGE_PATH: str = "/shared/data"

    TIDB_HOST: str = os.getenv("TIDB_HOST")
    TIDB_PORT: int = os.getenv("TIDB_PORT")
    TIDB_USER: str = os.getenv("TIDB_USERNAME")
    TIDB_PASSWORD: str = os.getenv("TIDB_PASSWORD")
    TIDB_DATABASE: str = os.getenv("TIDB_DATABASE")
    TIDB_SSL: bool = True

settings = Settings()