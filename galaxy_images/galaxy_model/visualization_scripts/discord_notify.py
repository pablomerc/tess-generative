"""Tiny Discord webhook helper. notify() never raises — Discord outages must not fail the job."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Optional

import requests


def notify(webhook_url: Optional[str], message: str, file_path: Optional[str | Path] = None) -> None:
    if not webhook_url:
        return
    try:
        if file_path is not None:
            p = Path(file_path)
            with open(p, "rb") as fh:
                files = {"file": (p.name, fh, "application/octet-stream")}
                data = {"content": message[:1900]}
                r = requests.post(webhook_url, data=data, files=files, timeout=30)
        else:
            r = requests.post(webhook_url, json={"content": message[:1900]}, timeout=15)
        if r.status_code >= 300:
            print(f"[discord_notify] non-2xx ({r.status_code}): {r.text[:200]}", file=sys.stderr)
    except Exception as e:
        print(f"[discord_notify] failed: {e}", file=sys.stderr)
