from app.platforms.botapi import bale, telegram
from app.platforms.digikala import digikala
from app.platforms.instagram import instagram

ADAPTERS = {a.platform: a for a in (telegram, bale, digikala, instagram)}

PLATFORM_INFO = {
    "telegram": {"label": "تلگرام", "connect": "bot", "can_publish": True, "can_crawl": True},
    "bale": {"label": "بله", "connect": "bot", "can_publish": True, "can_crawl": True},
    "digikala": {"label": "دیجی‌کالا", "connect": "otp", "can_publish": True, "can_crawl": True},
    "instagram": {"label": "اینستاگرام", "connect": "token", "can_publish": True, "can_crawl": True, "flag": True},
}


def get_adapter(platform: str):
    return ADAPTERS[platform]
