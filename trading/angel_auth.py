"""
Angel One SmartAPI Authentication Module
FREE — No monthly subscription needed. Just an Angel One trading account.

Register at: https://smartapi.angelbroking.com/
"""

import os
import json
import pyotp
import logging
from datetime import date
from pathlib import Path
from typing import Optional
from dotenv import load_dotenv

load_dotenv(override=True)

logger = logging.getLogger(__name__)

SESSION_FILE    = Path(__file__).parent / "session.json"
ANGEL_API_KEY   = os.getenv("ANGEL_API_KEY", "")
ANGEL_CLIENT_ID = os.getenv("ANGEL_CLIENT_ID", "")
ANGEL_PASSWORD  = os.getenv("ANGEL_PASSWORD", "")
ANGEL_TOTP_SECRET = os.getenv("ANGEL_TOTP_SECRET", "")


def _load_session() -> Optional[dict]:
    if SESSION_FILE.exists():
        with open(SESSION_FILE) as f:
            return json.load(f)
    return None


def _save_session(data: dict) -> None:
    with open(SESSION_FILE, "w") as f:
        json.dump(data, f, indent=2)


def _is_session_valid(session: Optional[dict]) -> bool:
    if not session:
        return False
    return session.get("login_date") == str(date.today())


def get_smartapi_client(force_refresh: bool = False):
    """
    Return an authenticated SmartConnect instance.
    Auto-generates a fresh session daily using TOTP.

    Returns:
        Tuple of (SmartConnect client, auth_token, feed_token)
    """
    if not ANGEL_API_KEY:
        raise ValueError(
            "ANGEL_API_KEY not set in .env!\n"
            "Register FREE at: https://smartapi.angelbroking.com/"
        )

    session = _load_session()

    # Reuse today's session
    if not force_refresh and _is_session_valid(session):
        try:
            from SmartApi import SmartConnect
            obj = SmartConnect(api_key=ANGEL_API_KEY)
            obj.setSessionExpiryHook(lambda: None)
            logger.info("Reusing cached Angel One session.")
            return obj, session["auth_token"], session["feed_token"]
        except Exception:
            pass  # Fall through to fresh login

    # Fresh login
    if not all([ANGEL_CLIENT_ID, ANGEL_PASSWORD, ANGEL_TOTP_SECRET]):
        raise ValueError(
            "Missing Angel One credentials in .env!\n"
            "Please set: ANGEL_CLIENT_ID, ANGEL_PASSWORD, ANGEL_TOTP_SECRET"
        )

    logger.info("Performing fresh Angel One SmartAPI login...")
    try:
        from SmartApi import SmartConnect

        obj = SmartConnect(api_key=ANGEL_API_KEY)
        totp = pyotp.TOTP(ANGEL_TOTP_SECRET).now()

        data = obj.generateSession(
            clientCode=ANGEL_CLIENT_ID,
            password=ANGEL_PASSWORD,
            totp=totp,
        )

        if data.get("status") is False:
            raise RuntimeError(f"Angel One login failed: {data.get('message')}")

        auth_token  = data["data"]["jwtToken"]
        feed_token  = obj.getfeedToken()
        refresh_tok = data["data"]["refreshToken"]

        _save_session({
            "login_date":    str(date.today()),
            "auth_token":    auth_token,
            "feed_token":    feed_token,
            "refresh_token": refresh_tok,
            "client_id":     ANGEL_CLIENT_ID,
        })

        logger.info(f"Angel One login successful for {ANGEL_CLIENT_ID}.")
        return obj, auth_token, feed_token

    except Exception as e:
        raise RuntimeError(f"Angel One SmartAPI authentication failed: {e}") from e


def get_client():
    """Convenience wrapper — returns just the SmartConnect object."""
    obj, _, _ = get_smartapi_client()
    return obj
