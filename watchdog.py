#!/usr/bin/env python3
"""
Watchdog — Independent trading-bot health monitor
==================================================
Runs on a schedule (e.g., every 15 min via launchd/cron) independently of
the main trading process. Alerts via Discord webhook if the bot hasn't run
within the expected window during market hours.

Alert logic
-----------
- Only fires during US market hours (Mon-Fri, 09:30–16:15 ET)
- Threshold: 45 min since last confirmed bot run
- Re-alert throttle: 2 h (avoids alert spam during extended outage)
- All Discord delivery via raw webhook — no dependency on the trading process

Dead-man's switch
-----------------
NOT handled here. The trading process pings an external health-check service
on every successful execution. This script handles the local alert only
(machine is running but bot is silent).

Configuration
-------------
Reads from .env in the same directory. Required key:
    DISCORD_WATCHDOG_WEBHOOK — Discord webhook URL for alerts

Optional:
    HEALTHCHECK_PING_URL — external dead-man's-switch service URL (e.g.,
                           healthchecks.io). Pinged on every successful
                           watchdog run when present.

Usage
-----
    python watchdog.py               # normal run
    python watchdog.py --force-fire  # bypass market-hours check (testing)
"""

import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from zoneinfo import ZoneInfo

# ── Config ────────────────────────────────────────────────────────────────────

BASE_DIR       = os.path.dirname(os.path.abspath(__file__))
POSITIONS_FILE = os.path.join(BASE_DIR, "positions.json")
STATE_FILE     = os.path.join(BASE_DIR, "watchdog_state.json")
ENV_FILE       = os.path.join(BASE_DIR, ".env")

ET = ZoneInfo("America/New_York")

THRESHOLD_MINUTES      = 45    # alert if no run for this long during market hours
RETHROTTLE_MINUTES     = 120   # minimum gap between repeated alerts
MARKET_OPEN_HOUR       = 9
MARKET_OPEN_MINUTE     = 30
MARKET_CLOSE_HOUR      = 16
MARKET_CLOSE_MINUTE    = 15    # 15 min grace after 4 PM close
MARKET_OPEN_GRACE_MINUTES = 45 # skip alerts in first 45 min after open

# ── Helpers ───────────────────────────────────────────────────────────────────

def load_env() -> dict:
    env: dict[str, str] = {}
    try:
        with open(ENV_FILE) as fh:
            for line in fh:
                line = line.strip()
                if line and not line.startswith("#") and "=" in line:
                    k, v = line.split("=", 1)
                    env[k.strip()] = v.strip()
    except Exception:
        pass
    return env


# Required .env keys for watchdog operation.
_WATCHDOG_REQUIRED_KEYS = ["DISCORD_WATCHDOG_WEBHOOK"]


def validate_env(env: dict) -> list[str]:
    """Return list of missing/placeholder keys. Empty list = OK."""
    bad = []
    for key in _WATCHDOG_REQUIRED_KEYS:
        val = env.get(key, "")
        if not val or val in ("REPLACE_ME", "***") or val.endswith("..."):
            bad.append(key)
    return bad


def is_market_hours(now: datetime) -> bool:
    """True if now is a weekday within US market hours (ET)."""
    et_now = now.astimezone(ET)
    if et_now.weekday() >= 5:          # Saturday=5, Sunday=6
        return False
    t = et_now.hour * 60 + et_now.minute
    market_open  = MARKET_OPEN_HOUR  * 60 + MARKET_OPEN_MINUTE
    market_close = MARKET_CLOSE_HOUR * 60 + MARKET_CLOSE_MINUTE
    return market_open <= t <= market_close


def load_watchdog_state() -> dict:
    try:
        return json.load(open(STATE_FILE))
    except Exception:
        return {"last_alert_sent": None}


def save_watchdog_state(state: dict) -> None:
    with open(STATE_FILE, "w") as fh:
        json.dump(state, fh, indent=2)


def get_last_bot_run() -> datetime | None:
    """Read last_bot_run timestamp from positions.json (written by the bot)."""
    try:
        data = json.load(open(POSITIONS_FILE))
        ts = data.get("last_bot_run")
        if ts:
            dt = datetime.fromisoformat(ts)
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=ET)
            return dt
    except Exception:
        pass
    return None


def send_webhook_alert(webhook_url: str, message: str) -> bool:
    """Post to Discord webhook via curl."""
    payload = json.dumps({"content": message})
    try:
        result = subprocess.run(
            [
                "curl", "-sS", "-o", "/dev/null", "-w", "%{http_code}",
                "-X", "POST", webhook_url,
                "-H", "Content-Type: application/json",
                "-d", payload,
            ],
            capture_output=True, text=True, timeout=15,
        )
        code = result.stdout.strip()
        if code in ("200", "204"):
            return True
        print(f"[watchdog] webhook HTTP {code}: {result.stderr[:200]}", file=sys.stderr)
        return False
    except Exception as exc:
        print(f"[watchdog] webhook delivery failed: {exc}", file=sys.stderr)
        return False


def ping_healthcheck(env: dict) -> None:
    """Ping an external health-check service if configured (optional)."""
    url = env.get("HEALTHCHECK_PING_URL", "").strip()
    if not url:
        return
    try:
        subprocess.run(
            ["curl", "-sS", "-o", "/dev/null", "--max-time", "10", url],
            check=False,
        )
    except Exception:
        pass


# ── Main ─────────────────────────────────────────────────────────────────────

def main() -> None:
    now = datetime.now(tz=timezone.utc)

    # Config validation — fail loudly before doing anything.
    # Intentionally NOT pinging the health-check service when config is invalid;
    # this causes the external dead-man's switch to fire after its grace period.
    env_early = load_env()
    missing   = validate_env(env_early)
    if missing:
        print(
            f"[watchdog] STARTUP ERROR — missing/placeholder .env keys: {', '.join(missing)}\n"
            f"  Edit {ENV_FILE} and replace REPLACE_ME values with real credentials.\n"
            f"  Health-check ping intentionally suppressed until config is valid.",
            file=sys.stderr,
        )
        sys.exit(2)

    # Skip outside market hours (bypass with --force-fire for testing only).
    force_fire = "--force-fire" in sys.argv or os.environ.get("WATCHDOG_FORCE_FIRE") == "1"
    if not force_fire and not is_market_hours(now):
        print(f"[watchdog] {now.astimezone(ET).strftime('%H:%M ET')} — outside market hours, skipping")
        return
    if force_fire:
        print("[watchdog] --force-fire: bypassing market-hours check")

    last_run = get_last_bot_run()
    now_et   = now.astimezone(ET)

    if last_run is None:
        elapsed_min = None
        elapsed_str = "NEVER"
    else:
        elapsed_min = (now - last_run).total_seconds() / 60
        elapsed_str = f"{elapsed_min:.0f}m ago ({last_run.astimezone(ET).strftime('%H:%M ET')})"

    print(f"[watchdog] {now_et.strftime('%H:%M ET')} | last_bot_run: {elapsed_str}")

    # Not overdue — nothing to do.
    if elapsed_min is not None and elapsed_min < THRESHOLD_MINUTES:
        print(f"[watchdog] OK — within {THRESHOLD_MINUTES}m threshold")
        env = load_env()
        ping_healthcheck(env)
        return

    # Market-open grace period: bot may start slightly after market open.
    market_open_today = now_et.replace(
        hour=MARKET_OPEN_HOUR, minute=MARKET_OPEN_MINUTE, second=0, microsecond=0
    )
    minutes_since_open = (now_et - market_open_today).total_seconds() / 60
    if 0 <= minutes_since_open < MARKET_OPEN_GRACE_MINUTES:
        print(
            f"[watchdog] Market-open grace period "
            f"({minutes_since_open:.0f}m since open, grace={MARKET_OPEN_GRACE_MINUTES}m) — skipping"
        )
        return

    # Overdue — check throttle before alerting.
    ws = load_watchdog_state()
    last_alert = ws.get("last_alert_sent")
    if last_alert:
        try:
            last_alert_dt = datetime.fromisoformat(last_alert)
            if last_alert_dt.tzinfo is None:
                last_alert_dt = last_alert_dt.replace(tzinfo=timezone.utc)
            since_alert = (now - last_alert_dt).total_seconds() / 60
            if since_alert < RETHROTTLE_MINUTES:
                print(
                    f"[watchdog] OVERDUE but throttled — alerted {since_alert:.0f}m ago, "
                    f"next in {RETHROTTLE_MINUTES - since_alert:.0f}m"
                )
                return
        except Exception:
            pass

    # Fire alert.
    env = load_env()
    webhook_url = env.get("DISCORD_WATCHDOG_WEBHOOK")
    if not webhook_url:
        print("[watchdog] ERROR: DISCORD_WATCHDOG_WEBHOOK not set in .env", file=sys.stderr)
        return

    msg = (
        f"⚠️ **Trading bot DOWN** — bot has not run for **{elapsed_str}**\n"
        f"Expected cadence: every 30 min during market hours\n"
        f"Action: check the trading process and agent auth status"
    )

    ok = send_webhook_alert(webhook_url, msg)
    if ok:
        ws["last_alert_sent"] = now.isoformat()
        save_watchdog_state(ws)
        print(f"[watchdog] 🚨 ALERT SENT — bot overdue by {elapsed_str}")
    else:
        print("[watchdog] Alert send FAILED", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
