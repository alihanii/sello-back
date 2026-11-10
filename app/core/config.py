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
    # Images: "local" (files in media_dir) or "s3" (any S3-compatible storage: AWS, ArvanCloud, MinIO...).
    storage: str = "local"
    s3_endpoint_url: str = ""  # e.g. https://s3.ir-thr-at1.arvanstorage.ir; empty for AWS
    s3_region: str = ""
    s3_bucket: str = ""
    s3_access_key: str = ""
    s3_secret_key: str = ""
    s3_prefix: str = ""  # optional folder inside the bucket
    s3_addressing_style: str = "auto"  # auto | path | virtual
    s3_acl: str = ""  # e.g. public-read when the provider needs it per object
    # Public base address of the bucket (e.g. https://mybucket.s3.ir-thr-at1.arvanstorage.ir) when it is public;
    # empty means a private bucket, and images are served through short-lived signed links.
    s3_public_url: str = ""
    s3_url_expire_seconds: int = 6 * 3600
    # SQLite file for az-iranian-bank-gateways' own table (the library runs on a headless Django).
    bank_db_path: str = "./bank.db"

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

    # Sales chat: "polling" (default; works without a public address) or "webhook" (needs public_base_url over
    # https; required when more than one backend instance runs).
    chat_transport: str = "polling"
    # Customer messages sent within this many seconds are answered together, in one model call.
    chat_debounce_seconds: float = 3.0

    # Smart proxies (admin panel › proxies): v2ray links run through a local Xray core.
    xray_path: str = "xray"  # binary name on PATH or a full path (xray.exe on Windows)
    xray_dir: str = "./xray"  # generated config and log
    proxy_check_seconds: float = 180

    # Store websites (storefront app): each store gets <subdomain>.<site_root_domain>; a wildcard DNS record
    # (*.root) must point to the server. site_server_ip is shown to sellers who connect their own domain.
    site_root_domain: str = ""  # e.g. sello.ir; empty = sites only on their own domains (or *.localhost in dev)
    site_scheme: str = "https"
    site_server_ip: str = ""

    worker_enabled: bool = True
    worker_concurrency: int = 4
    worker_poll_seconds: float = 1.0


@lru_cache
def get_settings() -> Settings:
    return Settings()


settings = get_settings()
