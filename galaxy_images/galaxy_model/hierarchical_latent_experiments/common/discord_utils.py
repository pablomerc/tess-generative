"""Thin wrappers around the Discord webhook used for status + plot delivery."""

from __future__ import annotations

import io
import os
from pathlib import Path

import requests

DEFAULT_WEBHOOK = os.environ.get(
    "DISCORD_WEBHOOK",
    "https://discord.com/api/webhooks/1499879903464788150/"
    "EiosG8MxzpVp4aGsb9FkxV7J-xEpZMABK4yTiwSBdEZ5GA7onfIKvBtHJOaPGli8SGv5",
)


def post_text(message: str, webhook: str = DEFAULT_WEBHOOK, timeout: float = 30.0) -> bool:
    try:
        resp = requests.post(webhook, json={"content": message[:1900]}, timeout=timeout)
        ok = resp.status_code in (200, 204)
        if not ok:
            print(f"[discord] text POST {resp.status_code}: {resp.text[:200]}")
        return ok
    except Exception as exc:
        print(f"[discord] text POST failed: {exc}")
        return False


def post_image(file_path: str | Path, message: str = "", webhook: str = DEFAULT_WEBHOOK,
               timeout: float = 60.0) -> bool:
    file_path = Path(file_path)
    if not file_path.exists():
        return post_text(f"{message} (file not found: {file_path})", webhook=webhook)
    try:
        with open(file_path, "rb") as f:
            data = f.read()
        resp = requests.post(
            webhook,
            data={"content": message[:1900]} if message else {},
            files={"file": (file_path.name, io.BytesIO(data), "image/png")},
            timeout=timeout,
        )
        ok = resp.status_code in (200, 204)
        if not ok:
            print(f"[discord] image POST {resp.status_code}: {resp.text[:200]}")
        return ok
    except Exception as exc:
        print(f"[discord] image POST failed: {exc}")
        return False
