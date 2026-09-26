"""
Zerodha Kite Connect Authentication Module
Handles automated daily login using TOTP and session management.
"""

import os
import json
import time
import pyotp
import logging
from datetime import datetime, date
from pathlib import Path
from typing import Optional
from dotenv import load_dotenv
from kiteconnect import KiteConnect

load_dotenv(override=True)

logger = logging.getLogger(__name__)

SESSION_FILE = Path(__file__).parent / "session.json"

KITE_API_KEY    = os.getenv("KITE_API_KEY", "")
KITE_API_SECRET = os.getenv("KITE_API_SECRET", "")
KITE_USER_ID    = os.getenv("KITE_USER_ID", "")
KITE_PASSWORD   = os.getenv("KITE_PASSWORD", "")
KITE_TOTP_SECRET = os.getenv("KITE_TOTP_SECRET", "")


def _load_session() -> Optional[dict]:
    """Load saved session from disk."""
    if SESSION_FILE.exists():
        with open(SESSION_FILE, "r") as f:
            return json.load(f)
    return None


def _save_session(data: dict) -> None:
    """Persist session data to disk."""
    with open(SESSION_FILE, "w") as f:
        json.dump(data, f, indent=2)


def _is_session_valid(session: Optional[dict]) -> bool:
    """Check if saved session was created today (tokens expire at midnight)."""
    if not session:
        return False
    session_date = session.get("login_date", "")
    return session_date == str(date.today())


def _automated_login() -> str:
    """
    Perform automated Zerodha login using credentials + TOTP.
    Returns a fresh request_token by driving the login flow via requests.
    """
    import requests
    from urllib.parse import urlparse, parse_qs

    if not all([KITE_API_KEY, KITE_USER_ID, KITE_PASSWORD, KITE_TOTP_SECRET]):
        raise ValueError(
            "Missing Zerodha credentials in .env file.\n"
            "Please set: KITE_API_KEY, KITE_API_SECRET, KITE_USER_ID, KITE_PASSWORD, KITE_TOTP_SECRET"
        )

    kite = KiteConnect(api_key=KITE_API_KEY)
    session = requests.Session()

    # Step 1: GET login page to get CSRF / session cookie
    login_url = "https://kite.zerodha.com/api/login"
    headers = {
        "Content-Type": "application/x-www-form-urlencoded",
        "User-Agent": "Mozilla/5.0",
        "x-kite-version": "3",
    }

    # Step 2: POST credentials
    payload = {"user_id": KITE_USER_ID, "password": KITE_PASSWORD}
    resp = session.post(login_url, data=payload, headers=headers)
    resp.raise_for_status()
    data = resp.json()

    if data.get("status") != "success":
        raise RuntimeError(f"Zerodha login step 1 failed: {data}")

    request_id = data["data"]["request_id"]

    # Step 3: POST TOTP for 2FA
    totp = pyotp.TOTP(KITE_TOTP_SECRET)
    twofa_url = "https://kite.zerodha.com/api/twofa"
    twofa_payload = {
        "user_id": KITE_USER_ID,
        "request_id": request_id,
        "twofa_value": totp.now(),
        "skip_totp": "false",
    }
    resp2 = session.post(twofa_url, data=twofa_payload, headers=headers)
    resp2.raise_for_status()
    data2 = resp2.json()

    if data2.get("status") != "success":
        raise RuntimeError(f"Zerodha 2FA failed: {data2}")

    # Step 4: Follow redirect to Kite Connect app to get request_token
    login_redirect = kite.login_url()
    resp3 = session.get(login_redirect, allow_redirects=True)
    final_url = resp3.url

    parsed = urlparse(final_url)
    params = parse_qs(parsed.query)
    request_token = params.get("request_token", [None])[0]

    if not request_token:
        raise RuntimeError(
            f"Could not extract request_token from redirect URL: {final_url}\n"
            "Ensure your Kite Connect app's redirect URL is set to 'http://localhost'."
        )

    return request_token


def get_kite_client(force_refresh: bool = False) -> KiteConnect:
    """
    Return an authenticated KiteConnect instance.
    Reuses today's session if available, otherwise performs automated login.

    Args:
        force_refresh: If True, forces a new login even if today's session exists.

    Returns:
        Authenticated KiteConnect client.
    """
    if not KITE_API_KEY:
        raise ValueError(
            "KITE_API_KEY not set in .env. Please add your Zerodha Kite Connect API key."
        )

    session = _load_session()

    if not force_refresh and _is_session_valid(session):
        kite = KiteConnect(api_key=KITE_API_KEY)
        kite.set_access_token(session["access_token"])
        logger.info("Using cached Kite session from today.")
        return kite

    logger.info("Performing fresh Zerodha login...")
    try:
        request_token = _automated_login()
        kite = KiteConnect(api_key=KITE_API_KEY)
        data = kite.generate_session(request_token, api_secret=KITE_API_SECRET)
        kite.set_access_token(data["access_token"])

        _save_session({
            "access_token": data["access_token"],
            "login_date": str(date.today()),
            "user_id": KITE_USER_ID,
        })

        logger.info(f"Kite login successful. Access token saved for {date.today()}.")
        return kite

    except Exception as e:
        raise RuntimeError(f"Kite Connect authentication failed: {e}") from e


def manual_token_login(access_token: str) -> KiteConnect:
    """
    Manually set an access token (useful when you've already logged in
    via the browser and have the token from Zerodha's redirect URL).
    """
    kite = KiteConnect(api_key=KITE_API_KEY)
    kite.set_access_token(access_token)
    _save_session({
        "access_token": access_token,
        "login_date": str(date.today()),
        "user_id": KITE_USER_ID,
    })
    logger.info("Access token set manually and saved.")
    return kite
