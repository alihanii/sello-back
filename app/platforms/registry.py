from app.platforms.botapi import bale, telegram
from app.platforms.digikala import digikala
from app.platforms.instagram import instagram
from app.platforms.linkedin import linkedin

ADAPTERS = {a.platform: a for a in (telegram, bale, digikala, instagram, linkedin)}

PLATFORM_INFO = {
    "telegram": {"label": "تلگرام", "connect": "bot", "can_publish": True, "can_crawl": True},
    "bale": {"label": "بله", "connect": "bot", "can_publish": True, "can_crawl": True},
    "digikala": {"label": "دیجی‌کالا", "connect": "otp", "can_publish": True, "can_crawl": False, "orders": True},
    "instagram": {"label": "اینستاگرام", "connect": "token", "can_publish": True, "can_crawl": True, "flag": True},
    # Email + password (or the browser's li_at cookie): posts with photos, and the assistant answers DMs and comments.
    "linkedin": {"label": "لینکدین", "connect": "login", "can_publish": True, "can_crawl": False, "flag": True},
}


def get_adapter(platform: str):
    return ADAPTERS[platform]
