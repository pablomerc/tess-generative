"""Discord webhook notifications for the AE baseline runs.

The webhook URL is NEVER committed: it is read from $DISCORD_WEBHOOK_URL or,
failing that, from ~/.config/discord_webhook (chmod 600). All senders swallow
errors — a dead webhook must never kill a training job.
"""

from __future__ import annotations

import os


def _webhook_url() -> str | None:
    url = os.environ.get("DISCORD_WEBHOOK_URL")
    if url:
        return url.strip()
    path = os.path.expanduser("~/.config/discord_webhook")
    if os.path.exists(path):
        with open(path) as f:
            return f.read().strip()
    return None


def discord_post(text: str, file_paths: list[str] | None = None) -> bool:
    url = _webhook_url()
    if not url:
        return False
    try:
        import requests
        files = {}
        handles = []
        for i, p in enumerate(file_paths or []):
            fh = open(p, "rb")
            handles.append(fh)
            files[f"file{i}"] = (os.path.basename(p), fh)
        resp = requests.post(url, data={"content": text[:1990]},
                             files=files or None, timeout=30)
        for fh in handles:
            fh.close()
        return resp.status_code in (200, 204)
    except Exception as e:  # noqa: BLE001 — never take down the job
        print(f"[notify] discord post failed: {e}")
        return False
