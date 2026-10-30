import os
import tempfile

_tmp = tempfile.mkdtemp()
os.environ["SELLO_DATABASE_URL"] = f"sqlite+aiosqlite:///{_tmp}/test.db"
os.environ["SELLO_MEDIA_DIR"] = f"{_tmp}/media"
os.environ["SELLO_WORKER_ENABLED"] = "false"
os.environ["SELLO_BANK_DB_PATH"] = f"{_tmp}/bank.db"
os.environ["SELLO_ADMIN_PHONES"] = '["09120000000"]'
os.environ["SELLO_OTP_MAX_SENDS_PER_10MIN"] = "100"

import httpx  # noqa: E402

from app.services import ai as _ai  # noqa: E402

_ai.BUSY_RETRY_DELAYS = (0.0, 0.0)
import pytest  # noqa: E402

from app.main import app  # noqa: E402
from app.core.db import init_db  # noqa: E402
from app.seed import seed  # noqa: E402


@pytest.fixture(scope="session")
async def client():
    await init_db()
    await seed()
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test/api/v1") as c:
        yield c


async def login(client, phone: str) -> dict:
    r = await client.post("/auth/otp/send", json={"phone": phone})
    assert r.status_code == 200, r.text
    r = await client.post("/auth/otp/verify", json={"phone": phone, "code": r.json()["dev_code"]})
    assert r.status_code == 200, r.text
    return {"Authorization": f"Bearer {r.json()['access_token']}"}


async def drain():
    """Run every queued operation, like the worker would."""
    from app.services.worker import _claim, run_operation

    while (op_id := await _claim()) is not None:
        await run_operation(op_id)
