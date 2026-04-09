"""Send a Discord webhook notification."""

from __future__ import annotations

import argparse
import json
import os
import sys
import urllib.error
import urllib.request
from datetime import datetime


def main() -> int:
    parser = argparse.ArgumentParser(description="Send a Discord webhook message.")
    parser.add_argument("--message", default=None, help="Message content.")
    parser.add_argument("--everyone", action="store_true", help="Mention @everyone.")
    parser.add_argument(
        "--tail-file",
        default=None,
        help="Append the last N lines of this file to the message.",
    )
    parser.add_argument(
        "--tail-lines",
        type=int,
        default=5,
        help="Number of lines to append when using --tail-file (default: 5).",
    )
    parser.add_argument(
        "--webhook",
        default="https://discord.com/api/webhooks/1466378846340255872/o_EzwqnCllSZYk8p3IhZ9aAUqWwLxlIDSD6XkRPUfmvcsrSs2Ri89VIrI5Umv9RKVtv7",
        help="Discord webhook URL.",
    )
    args = parser.parse_args()

    webhook_url = args.webhook or os.environ.get("DISCORD_WEBHOOK_URL")

    message = args.message
    if message is None:
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        message = f"Job termine a {timestamp}"
    if args.everyone and "@everyone" not in message:
        message = f"@everyone {message}"

    if args.tail_file:
        tail_lines = max(args.tail_lines, 0)
        tail_text = ""
        try:
            with open(args.tail_file, "r", encoding="utf-8", errors="replace") as handle:
                if tail_lines == 0:
                    tail_text = ""
                else:
                    lines = handle.readlines()
                    tail_text = "".join(lines[-tail_lines:])
        except OSError as exc:
            print(f"Tail read failed: {exc}", file=sys.stderr)
            return 1

        if tail_text:
            message = f"{message}\n\n```\n{tail_text}```"

    payload = {"content": message}
    if args.everyone:
        payload["allowed_mentions"] = {"parse": ["everyone"]}
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        webhook_url,
        data=data,
        headers={
            "Content-Type": "application/json",
            "User-Agent": "curl/7.79.1",
        },
    )
    try:
        with urllib.request.urlopen(req, timeout=10) as resp:
            if resp.status >= 400:
                print(f"Webhook failed: HTTP {resp.status}", file=sys.stderr)
                return 1
    except urllib.error.HTTPError as exc:
        body = ""
        try:
            body = exc.read().decode("utf-8", errors="replace")
        except Exception:
            body = ""
        if body:
            print(f"Webhook failed: HTTP {exc.code} {body}", file=sys.stderr)
        else:
            print(f"Webhook failed: HTTP {exc.code}", file=sys.stderr)
        return 1
    except urllib.error.URLError as exc:
        print(f"Webhook failed: {exc.reason}", file=sys.stderr)
        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
