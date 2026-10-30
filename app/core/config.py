from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_prefix="SELLO_", extra="ignore")

    env: str = "dev"
    database_url: str = "sqlite+aiosqlite:///./sello.db"
    jwt_secret: str = "dev-secret-change-me"
    jwt_access_minutes: int = 60 * 24
    # Fernet key for platform credentials; generated per-process in dev if empty.
    credentials_key: str = ""
    cors_origins: list[str] = ["http://localhost:3000"]
    public_base_url: str = "http://localhost:8000"
    # Where the seller app runs; OAuth redirects (Instagram) come back here.
    frontend_base_url: str = "http://localhost:3000"
    media_dir: str = "./media"

    # OTP: in dev the code is returned in the response and logged instead of sent by SMS.
    otp_dev_mode: bool = True
    otp_ttl_seconds: int = 120
    otp_max_sends_per_10min: int = 3
    sms_provider: str = ""  # kavenegar | smsir
    sms_api_key: str = ""

    # Phones listed here get the admin role on first login.
    admin_phones: list[str] = ["09120000000"]

    # AI keys, models and routing are managed from the admin panel (ai_providers table).

    # Instagram: the Meta app id/secret and an optional proxy are set in the admin panel; these are fallbacks.
    instagram_enabled: bool = False
    instagram_graph_version: str = "v25.0"
    instagram_app_id: str = ""
    instagram_app_secret: str = ""
    instagram_proxy: str = ""

    worker_enabled: bool = True
    worker_concurrency: int = 4
    worker_poll_seconds: float = 1.0


@lru_cache
def get_settings() -> Settings:
    return Settings()


settings = get_settings()
