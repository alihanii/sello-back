"""Live chat updates in this process: a web chat page waits on its conversation instead of asking the server every few
seconds. `add_message` marks the conversation on the session; once that session commits, everyone waiting on it is
woken and reads the new rows. One backend process serves the API, so an in-memory list is enough."""

import asyncio
from collections import defaultdict
from contextlib import contextmanager

from sqlalchemy import event
from sqlalchemy.orm import Session

_waiting: dict[int, set[asyncio.Event]] = defaultdict(set)


def mark(session: Session, conv_id: int) -> None:
    session.info.setdefault("live_convs", set()).add(conv_id)


@event.listens_for(Session, "after_commit")
def _after_commit(session: Session) -> None:
    for cid in session.info.pop("live_convs", ()):
        for e in list(_waiting.get(cid, ())):
            e.set()


@event.listens_for(Session, "after_rollback")
def _after_rollback(session: Session) -> None:
    session.info.pop("live_convs", None)


@contextmanager
def watch(conv_id: int):
    e = asyncio.Event()
    _waiting[conv_id].add(e)
    try:
        yield e
    finally:
        _waiting[conv_id].discard(e)
        if not _waiting[conv_id]:
            _waiting.pop(conv_id, None)
