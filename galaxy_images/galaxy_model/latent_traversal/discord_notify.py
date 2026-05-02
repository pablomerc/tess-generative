"""Tiny Discord notify CLI: `python3 discord_notify.py "message"`."""
import sys
import requests

WEBHOOK = (
    "https://discord.com/api/webhooks/1497979386144493680/"
    "VA-xWhfTWzc-oeC5EvPzyqEk_MW52wZsK2RyLS0egfhHHHhBxrmb9NGawy0rIpfvn3Zo"
)


def notify(msg: str, timeout: float = 10.0) -> None:
    try:
        requests.post(WEBHOOK, json={"content": msg[:1900]}, timeout=timeout)
    except Exception as e:
        print(f"discord_notify failed: {e}", file=sys.stderr)


if __name__ == "__main__":
    notify(" ".join(sys.argv[1:]) if len(sys.argv) > 1 else "(empty notify)")
