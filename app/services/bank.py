"""Online plan payments through az-iranian-bank-gateways.

The library is a Django app (Django ORM models, settings, url reversing, Django request objects), so we
run a minimal, headless Django inside the FastAPI process: its own small database for the library's
`Bank` table, no Django views or middleware. Gateway credentials come from the admin panel through a
custom settings reader, so changes apply without a restart. Every library call is synchronous and
touches the Django ORM, so callers run them in a worker thread (`asyncio.to_thread`)."""

import threading
from typing import Any
from urllib.parse import urlparse

from app.core.config import settings

BANKS: dict[str, dict[str, Any]] = {
    "ZARINPAL": {"label": "زرین‌پال", "fields": ["MERCHANT_CODE"], "sandbox": True},
    "ZIBAL": {"label": "زیبال", "fields": ["MERCHANT_CODE"]},
    "IRANDARGAH": {"label": "ایران‌درگاه", "fields": ["MERCHANT_CODE"]},
    "BAHAMTA": {"label": "باهمتا", "fields": ["MERCHANT_CODE"]},
    "SEP": {"label": "سامان (سپ)", "fields": ["MERCHANT_CODE", "TERMINAL_CODE"]},
    "BMI": {"label": "ملی (سداد)", "fields": ["MERCHANT_CODE", "TERMINAL_CODE", "SECRET_KEY"]},
    "MELLAT": {"label": "ملت (به‌پرداخت)", "fields": ["TERMINAL_CODE", "USERNAME", "PASSWORD"]},
}

# Filled from the admin settings before each library call; read by SelloReader.
_runtime: dict[str, Any] = {"bank": "ZARINPAL", "gateways": {}}
_lock = threading.Lock()
_ready = False


def _setup() -> None:
    global _ready
    if _ready:
        return
    with _lock:
        if _ready:
            return
        import django
        from django.conf import settings as dj

        if not dj.configured:
            dj.configure(
                DEBUG=False,
                SECRET_KEY=settings.jwt_secret,
                ALLOWED_HOSTS=["*"],
                USE_TZ=True,
                TIME_ZONE="UTC",
                INSTALLED_APPS=["django.contrib.contenttypes", "azbankgateways"],
                DATABASES={"default": {"ENGINE": "django.db.backends.sqlite3", "NAME": settings.bank_db_path}},
                ROOT_URLCONF="app.services.bank_urls",
                DEFAULT_AUTO_FIELD="django.db.models.AutoField",
                SECURE_REFERRER_POLICY="same-origin",
                AZ_IRANIAN_BANK_GATEWAYS={
                    "SETTING_VALUE_READER_CLASS": "app.services.bank.SelloReader",
                    "CURRENCY": "IRT",  # our prices are in Toman; the library converts per gateway
                    "BANK_TIMEOUT": 15,
                    "IS_SAFE_GET_GATEWAY_PAYMENT": True,
                    "TRACKING_CODE_QUERY_PARAM": "tc",
                },
            )
        django.setup()
        from django.core.management import call_command

        call_command("migrate", "azbankgateways", verbosity=0, interactive=False)
        _ready = True


def _reader_base():
    from azbankgateways.readers import DefaultReader

    return DefaultReader


def __getattr__(name: str):
    # SelloReader is built lazily because importing the library requires Django to be configured first.
    if name == "SelloReader":
        base = _reader_base()

        class SelloReader(base):  # type: ignore[misc, valid-type]
            def read(self, bank_type, identifier):
                return dict(_runtime["gateways"].get(str(bank_type), {}))

            def default(self, identifier):
                return _runtime["bank"]

            def currency(self, identifier):
                return "IRT"

            def get_bank_priorities(self, identifier):
                return [_runtime["bank"]]

        globals()["SelloReader"] = SelloReader
        return SelloReader
    raise AttributeError(name)


def _request(path: str = "/", query: dict | None = None, data: dict | None = None, method: str = "GET"):
    """A Django request whose host is our public URL, so the library builds absolute callback URLs."""
    from django.test import RequestFactory

    base = urlparse(settings.public_base_url)
    rf = RequestFactory(HTTP_HOST=base.netloc, secure=base.scheme == "https")
    if method == "POST":
        from urllib.parse import urlencode

        return rf.post(f"{path}?{urlencode(query or {})}", data=data or {})
    return rf.get(path, query or {})


def configure(cfg: dict[str, Any]) -> None:
    gateways = {}
    for code, values in (cfg.get("gateways") or {}).items():
        if code not in BANKS:
            continue
        g = {f: values.get(f, "") for f in BANKS[code]["fields"]}
        if BANKS[code].get("sandbox"):
            g["SANDBOX"] = 1 if values.get("SANDBOX") else 0
        gateways[code] = g
    _runtime.update(bank=cfg.get("bank") or "ZARINPAL", gateways=gateways)


def start(cfg: dict[str, Any], *, amount_toman: int, mobile: str, return_url: str) -> dict[str, Any]:
    """Creates the gateway token. Returns tracking_code, reference_number and how to send the user to the bank."""
    _setup()
    configure(cfg)
    from azbankgateways.bankfactories import BankFactory

    bank = BankFactory().create(_runtime["bank"])
    bank.set_request(_request())
    bank.set_amount(amount_toman)
    bank.set_client_callback_url(return_url)
    if mobile:
        bank.set_mobile_number(mobile)
    record = bank.ready()
    gateway = bank.get_gateway()  # {"url", "params", "method"}
    return {"tracking_code": str(record.tracking_code), "reference_number": record.reference_number,
            "bank_type": record.bank_type, "gateway": gateway}


def verify(cfg: dict[str, Any], *, query: dict, form: dict, method: str) -> dict[str, Any]:
    """Runs the library's verify for a gateway callback and reports the outcome."""
    _setup()
    configure(cfg)
    from azbankgateways.bankfactories import BankFactory
    from azbankgateways.exceptions import AZBankGatewaysException
    from azbankgateways.models import Bank

    bank_type = query.get("bank_type") or _runtime["bank"]
    bank = BankFactory().create(bank_type, identifier=query.get("identifier") or "1")
    error = ""
    try:
        bank.verify_from_gateway(_request("/api/v1/payments/callback/", query, form, method))
    except AZBankGatewaysException as exc:
        error = str(exc) or exc.__class__.__name__
    tracking = bank.get_tracking_code()
    record = Bank.objects.filter(tracking_code=str(tracking)).first() if tracking else None
    return {"tracking_code": str(tracking or ""), "success": bool(record and record.is_success),
            "status": record.status if record else "", "message": (record.response_result if record else "") or error}
