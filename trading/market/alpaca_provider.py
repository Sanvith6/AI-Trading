"""
Alpaca Real-Time Market Data Provider & WebSocket Streaming Client.

Operational Production Market Data Feed for US Equities (Paper & Live).
- WebSocket streaming via wss://stream.data.alpaca.markets/v2/iex (or sip).
- REST fallback via https://data.alpaca.markets/v2/stocks.
- Offline/Research fallback to USMarketDataProvider (Yahoo) if keys are unset.
- Strategy-aware quote latency and age tracking (quote_age, provider_latency, received_age).
"""

import os
import json
import time
import asyncio
import logging
import threading
from typing import Dict, Any, List, Optional, Tuple
from urllib.request import Request, urlopen
from urllib.error import URLError, HTTPError
import pandas as pd

from .interface import MarketDataProvider, MarketQuote
from .yahoo_provider import USMarketDataProvider

logger = logging.getLogger(__name__)

DEFAULT_DATA_URL = "https://data.alpaca.markets/v2"
DEFAULT_WS_URL = "wss://stream.data.alpaca.markets/v2/iex"


class AlpacaMarketDataProvider(MarketDataProvider):
    """
    Production real-time US market data provider powered by Alpaca's Market Data API & WebSocket Stream.
    Maintains an ultra-low-latency in-memory cache updated by streaming events.
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        secret_key: Optional[str] = None,
        base_url: Optional[str] = None,
        feed: str = "iex",  # "iex" (standard/free) or "sip" (full tape)
        fallback_provider: Optional[MarketDataProvider] = None,
        allow_research_fallback: Optional[bool] = None,
    ):
        self.api_key = (
            api_key
            or os.getenv("ALPACA_DATA_API_KEY")
            or os.getenv("ALPACA_PAPER_API_KEY")
            or os.getenv("ALPACA_LIVE_API_KEY")
            or os.getenv("APCA_API_KEY_ID", "")
        )
        self.secret_key = (
            secret_key
            or os.getenv("ALPACA_DATA_SECRET_KEY")
            or os.getenv("ALPACA_PAPER_SECRET_KEY")
            or os.getenv("ALPACA_LIVE_SECRET_KEY")
            or os.getenv("APCA_API_SECRET_KEY", "")
        )
        self.base_url = (base_url or os.getenv("ALPACA_DATA_URL", DEFAULT_DATA_URL)).rstrip("/")
        self.feed = feed or os.getenv("ALPACA_DATA_FEED", "iex")
        self.fallback_provider = fallback_provider or USMarketDataProvider()

        # Operational fallback policy: if STRICT_MARKET_FEED=1, disable silent fallback
        if allow_research_fallback is not None:
            self.allow_research_fallback = allow_research_fallback
        else:
            strict_flag = os.getenv("STRICT_MARKET_FEED", "0").strip().lower() in ("1", "true", "yes")
            self.allow_research_fallback = not strict_flag

        # In-memory real-time quote and bar cache updated via stream or REST
        self._quote_cache: Dict[str, MarketQuote] = {}
        self._cache_lock = threading.Lock()

        # Streaming thread control & operational diagnostics
        self._streaming_active = False
        self._streaming_thread: Optional[threading.Thread] = None
        self._subscribed_symbols: List[str] = []
        self._fallback_active: bool = False
        self._reconnect_count: int = 0
        self._last_message_time: float = 0.0
        self._last_provider_latency: float = 0.0
        self._last_quote_age: float = 0.0
        self._last_received_age: float = 0.0
        self._connection_state: str = "HEALTHY (ONLINE)" if self.has_credentials else "FALLBACK_RESEARCH (Yahoo - Awaiting Alpaca Keys)"

    @property
    def has_credentials(self) -> bool:
        return bool(self.api_key and self.secret_key and not self.api_key.startswith("<"))

    @property
    def fallback_active(self) -> bool:
        return self._fallback_active

    @property
    def reconnect_count(self) -> int:
        return self._reconnect_count

    @property
    def last_message_age(self) -> float:
        if self._last_message_time > 0:
            return round(max(0.0, time.time() - self._last_message_time), 2)
        return 0.0

    @property
    def subscribed_symbols(self) -> List[str]:
        return list(self._subscribed_symbols)

    @property
    def connection_state(self) -> str:
        return self._connection_state

    @property
    def provider_latency(self) -> float:
        return self._last_provider_latency

    @property
    def quote_age(self) -> float:
        return self._last_quote_age

    @property
    def received_age(self) -> float:
        return self._last_received_age

    def _get_headers(self) -> Dict[str, str]:
        return {
            "APCA-API-KEY-ID": self.api_key,
            "APCA-API-SECRET-KEY": self.secret_key,
            "Content-Type": "application/json",
        }

    def get_quote(self, symbol: str) -> MarketQuote:
        """
        Retrieves current market quote.
        Prefers real-time streaming cache; falls back to REST or secondary research feed.
        """
        sym = symbol.strip().upper()
        now = time.time()

        # 1. Check in-memory streaming cache
        with self._cache_lock:
            cached = self._quote_cache.get(sym)
            if cached and (now - cached.received_at) < 5.0:
                self._last_provider_latency = cached.provider_latency_seconds
                self._last_quote_age = cached.quote_age_seconds
                self._last_received_age = cached.received_age_seconds
                return cached

        # 2. Query Alpaca REST Market Data API if credentials exist
        if self.has_credentials:
            try:
                t0 = time.perf_counter()
                url = f"{self.base_url}/stocks/{sym}/bars/latest?feed={self.feed}"
                req = Request(url, headers=self._get_headers())
                with urlopen(req, timeout=5) as resp:
                    data = json.loads(resp.read().decode("utf-8"))
                latency = round(time.perf_counter() - t0, 4)

                bar = data.get("bar")
                if bar:
                    ts_str = bar.get("t", time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(now)))
                    quote = MarketQuote(
                        symbol=sym,
                        last_price=float(bar.get("c", 0.0)),
                        open=float(bar.get("o", 0.0)),
                        high=float(bar.get("h", 0.0)),
                        low=float(bar.get("l", 0.0)),
                        close=float(bar.get("c", 0.0)),
                        volume=int(bar.get("v", 0)),
                        change_pct=round(((bar.get("c", 0.0) - bar.get("o", 0.0)) / bar.get("o", 1.0)) * 100.0, 2) if bar.get("o") else 0.0,
                        data_timestamp=ts_str,
                        received_at=now,
                        age_seconds=latency,
                        bar_interval="1m",
                        data_mode="real-time",
                        source=f"alpaca_{self.feed}",
                        quality_status="VALID",
                        provider_latency_seconds=latency,
                        received_age_seconds=0.0,
                    )
                    self._last_message_time = now
                    self._last_provider_latency = latency
                    self._last_quote_age = latency
                    self._last_received_age = 0.0
                    self._fallback_active = False
                    self._connection_state = "HEALTHY (ONLINE)"
                    with self._cache_lock:
                        self._quote_cache[sym] = quote
                    return quote
            except Exception as e:
                logger.warning("Alpaca Market Data REST query for %s failed (%s).", sym, e)

        # 3. Fallback or Degraded Halt
        if not self.allow_research_fallback:
            self._fallback_active = False
            self._connection_state = "DATA_DEGRADED (HALT NEW ORDERS)"
            logger.error("Alpaca operational market feed failed for %s. Research fallback prohibited during operational trading.", sym)
            degraded_quote = MarketQuote(
                symbol=sym,
                last_price=0.0,
                open=0.0,
                high=0.0,
                low=0.0,
                close=0.0,
                volume=0,
                change_pct=0.0,
                data_timestamp=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(now)),
                received_at=now,
                age_seconds=999.0,
                bar_interval="1m",
                data_mode="DEGRADED",
                source="alpaca_degraded",
                quality_status="DEGRADED",
                provider_latency_seconds=0.0,
                received_age_seconds=999.0,
            )
            return degraded_quote

        # Secondary research provider fallback (research/local mode only)
        self._fallback_active = True
        self._connection_state = "FALLBACK_RESEARCH (Yahoo)"
        fallback_quote = self.fallback_provider.get_quote(sym)
        self._last_provider_latency = fallback_quote.provider_latency_seconds
        self._last_quote_age = fallback_quote.quote_age_seconds
        self._last_received_age = fallback_quote.received_age_seconds
        with self._cache_lock:
            self._quote_cache[sym] = fallback_quote
        return fallback_quote

    def get_watchlist_quotes(self, symbols: List[str]) -> Dict[str, MarketQuote]:
        """
        Retrieves batch quotes across multiple symbols.
        Uses Alpaca multi-symbol bar snapshot when available for maximum speed.
        """
        if not symbols:
            return {}

        now = time.time()
        results: Dict[str, MarketQuote] = {}
        missing_symbols = []

        # Check streaming cache first
        with self._cache_lock:
            for s in symbols:
                sym = s.strip().upper()
                c = self._quote_cache.get(sym)
                if c and (now - c.received_at) < 5.0:
                    results[sym] = c
                else:
                    missing_symbols.append(sym)

        if not missing_symbols:
            return results

        # Batch REST query for missing symbols
        if self.has_credentials:
            try:
                t0 = time.perf_counter()
                sym_str = ",".join(missing_symbols[:50])
                url = f"{self.base_url}/stocks/bars/latest?symbols={sym_str}&feed={self.feed}"
                req = Request(url, headers=self._get_headers())
                with urlopen(req, timeout=8) as resp:
                    data = json.loads(resp.read().decode("utf-8"))
                latency = round(time.perf_counter() - t0, 4)

                bars_dict = data.get("bars", {})
                for sym, bar in bars_dict.items():
                    ts_str = bar.get("t", time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(now)))
                    q = MarketQuote(
                        symbol=sym,
                        last_price=float(bar.get("c", 0.0)),
                        open=float(bar.get("o", 0.0)),
                        high=float(bar.get("h", 0.0)),
                        low=float(bar.get("l", 0.0)),
                        close=float(bar.get("c", 0.0)),
                        volume=int(bar.get("v", 0)),
                        change_pct=round(((bar.get("c", 0.0) - bar.get("o", 0.0)) / bar.get("o", 1.0)) * 100.0, 2) if bar.get("o") else 0.0,
                        data_timestamp=ts_str,
                        received_at=now,
                        age_seconds=latency,
                        bar_interval="1m",
                        data_mode="real-time",
                        source=f"alpaca_{self.feed}",
                        quality_status="VALID",
                        provider_latency_seconds=latency,
                        received_age_seconds=0.0,
                    )
                    results[sym] = q
                    with self._cache_lock:
                        self._quote_cache[sym] = q
            except Exception as e:
                logger.warning("Alpaca batch market query failed (%s). Falling back to research feed.", e)

        # Populate any still-missing symbols via fallback provider if permitted
        remaining = [s for s in symbols if s not in results]
        if remaining:
            if not self.allow_research_fallback:
                self._fallback_active = False
                self._connection_state = "DATA_DEGRADED (HALT NEW ORDERS)"
                logger.error("Alpaca batch query missed %d symbols and research fallback is prohibited.", len(remaining))
                return results

            self._fallback_active = True
            self._connection_state = "FALLBACK_RESEARCH (Yahoo)"
            fb_quotes = self.fallback_provider.get_watchlist_quotes(remaining)
            results.update(fb_quotes)
            with self._cache_lock:
                self._quote_cache.update(fb_quotes)

        return results

    def get_bars(self, symbol: str, interval: str = "1d", limit: int = 120) -> pd.DataFrame:
        """
        Retrieves historical OHLCV bars.
        """
        sym = symbol.strip().upper()
        if self.has_credentials:
            try:
                tf = "1Day" if interval.lower() in ("1d", "1day", "d") else "1Min"
                url = f"{self.base_url}/stocks/{sym}/bars?timeframe={tf}&limit={limit}&feed={self.feed}"
                req = Request(url, headers=self._get_headers())
                with urlopen(req, timeout=10) as resp:
                    data = json.loads(resp.read().decode("utf-8"))
                bars = data.get("bars", [])
                if bars:
                    df = pd.DataFrame(bars)
                    df = df.rename(columns={"t": "timestamp", "o": "open", "h": "high", "l": "low", "c": "close", "v": "volume"})
                    df["timestamp"] = pd.to_datetime(df["timestamp"])
                    df = df.set_index("timestamp")
                    return df[["open", "high", "low", "close", "volume"]]
            except Exception as e:
                logger.warning("Alpaca get_bars failed for %s (%s). Falling back.", sym, e)

        return self.fallback_provider.get_bars(sym, interval=interval, limit=limit)

    def start_stream(self, symbols: List[str]) -> bool:
        """
        Initializes background WebSocket stream for real-time stock updates.
        """
        if not self.has_credentials:
            logger.info("Alpaca credentials missing; WebSocket streaming inactive (REST/fallback active).")
            return False

        if self._streaming_active:
            return True

        self._subscribed_symbols = symbols
        self._streaming_active = True
        self._streaming_thread = threading.Thread(target=self._run_stream_loop, daemon=True)
        self._streaming_thread.start()
        logger.info("Alpaca real-time WebSocket stream initiated for %d symbols.", len(symbols))
        return True

    def stop_stream(self) -> None:
        """Stops background WebSocket stream."""
        self._streaming_active = False

    def _run_stream_loop(self) -> None:
        """Background thread executing async WebSocket client."""
        asyncio.run(self._async_stream_client())

    async def _async_stream_client(self) -> None:
        """Async client handling WebSocket connection, authentication, and message dispatch."""
        try:
            import websockets
        except ImportError:
            logger.warning("websockets package not available; streaming listener disabled.")
            return

        ws_url = os.getenv("ALPACA_WS_URL", DEFAULT_WS_URL)
        while self._streaming_active:
            try:
                self._reconnect_count += 1
                async with websockets.connect(ws_url) as ws:
                    # 1. Receive connected message
                    greeting = await ws.recv()
                    self._last_message_time = time.time()
                    logger.debug("Alpaca WS connected: %s", greeting)

                    # 2. Authenticate
                    auth_msg = {
                        "action": "auth",
                        "key": self.api_key,
                        "secret": self.secret_key,
                    }
                    await ws.send(json.dumps(auth_msg))
                    auth_resp = await ws.recv()
                    self._last_message_time = time.time()
                    self._connection_state = "HEALTHY (ONLINE)"
                    logger.debug("Alpaca WS auth response: %s", auth_resp)

                    # 3. Subscribe to bars / trades for universe
                    sub_msg = {
                        "action": "subscribe",
                        "bars": self._subscribed_symbols,
                        "quotes": self._subscribed_symbols[:10],
                    }
                    await ws.send(json.dumps(sub_msg))

                    # 4. Message ingestion loop
                    while self._streaming_active:
                        msg_raw = await asyncio.wait_for(ws.recv(), timeout=30.0)
                        now = time.time()
                        self._last_message_time = now
                        msgs = json.loads(msg_raw)
                        for m in msgs:
                            msg_type = m.get("T")
                            if msg_type == "b":  # Bar
                                sym = m.get("S")
                                if sym:
                                    q = MarketQuote(
                                        symbol=sym,
                                        last_price=float(m.get("c", 0.0)),
                                        open=float(m.get("o", 0.0)),
                                        high=float(m.get("h", 0.0)),
                                        low=float(m.get("l", 0.0)),
                                        close=float(m.get("c", 0.0)),
                                        volume=int(m.get("v", 0)),
                                        change_pct=0.0,
                                        data_timestamp=m.get("t", ""),
                                        received_at=now,
                                        age_seconds=0.01,
                                        bar_interval="1m",
                                        data_mode="real-time-stream",
                                        source="alpaca_ws",
                                        quality_status="VALID",
                                        provider_latency_seconds=0.005,
                                        received_age_seconds=0.0,
                                    )
                                    with self._cache_lock:
                                        self._quote_cache[sym] = q
            except Exception as e:
                self._connection_state = "RECONNECTING"
                logger.debug("Alpaca WS stream connection cycle: %s", e)
                await asyncio.sleep(5.0)
