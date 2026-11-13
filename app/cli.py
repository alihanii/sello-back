"""Server commands. On the server:

    cd /opt/sello && docker compose exec -T backend python -m app.cli admin 09XXXXXXXXX
"""

import asyncio
import sys

from fastapi import HTTPException

from app.core.db import SessionLocal, init_db
from app.routers.auth import normalize_phone
from app.services.accounts import make_admin

USAGE = "usage: python -m app.cli admin <mobile number>   (creates the account, or makes an existing one an admin)"


async def _admin(raw: str) -> int:
    try:
        phone = normalize_phone(raw)
    except HTTPException:
        print(f"not a valid mobile number: {raw}\n{USAGE}")
        return 2
    await init_db()
    async with SessionLocal() as db:
        _, created = await make_admin(db, phone)
        await db.commit()
    print(f"{'admin account created' if created else 'existing account is now an admin'}: {phone}")
    print("Sign in with this number at the login page.")
    return 0


def main(argv: list[str]) -> int:
    if len(argv) == 2 and argv[0] == "admin":
        return asyncio.run(_admin(argv[1]))
    print(USAGE)
    return 2


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
