"""
Optional Alpaca adapter (paper trading by default).

Adapted from FinRL-X `src/trading/alpaca_manager.py` (Apache 2.0, AI4Finance Foundation), reduced to
what MarketIQ needs: account, positions, and market orders via the REST API. It is only constructed
when APCA_API_KEY / APCA_API_SECRET are set.

Safety: real-money trading is disabled by default. The adapter refuses to initialize against a
non-paper base URL unless ENABLE_LIVE_TRADING=true is set explicitly, and it logs loudly when it
does. MarketIQ's own paper broker (`marketiq.paper.broker`) needs no credentials and is the default.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

import requests

from marketiq.config.settings import get_settings

logger = logging.getLogger(__name__)


class LiveTradingDisabled(RuntimeError):
    pass


class AlpacaAdapter:
    def __init__(self, api_key: Optional[str] = None, api_secret: Optional[str] = None, base_url: Optional[str] = None,
                 session: Optional[requests.Session] = None):
        s = get_settings()
        a = s.alpaca
        self.api_key = api_key or (a.api_key.get_secret_value() if a.api_key else None)
        self.api_secret = api_secret or (a.api_secret.get_secret_value() if a.api_secret else None)
        self.base_url = (base_url or a.base_url).rstrip("/")
        if not self.api_key or not self.api_secret:
            raise RuntimeError("Alpaca credentials are not configured (APCA_API_KEY / APCA_API_SECRET)")
        self.is_paper = "paper-api" in self.base_url
        if not self.is_paper and not s.enable_live_trading:
            raise LiveTradingDisabled(
                "Refusing to connect to a live Alpaca endpoint: set ENABLE_LIVE_TRADING=true explicitly to allow real-money trading"
            )
        if not self.is_paper:
            logger.warning("LIVE TRADING ENABLED against %s. Real money can move.", self.base_url)
        self.http = session or requests.Session()
        self.http.headers.update({"APCA-API-KEY-ID": self.api_key, "APCA-API-SECRET-KEY": self.api_secret})

    def _request(self, method: str, path: str, **kwargs) -> Any:
        url = f"{self.base_url}/v2{path}"
        resp = self.http.request(method, url, timeout=30, **kwargs)
        resp.raise_for_status()
        return resp.json() if resp.content else None

    def account(self) -> Dict[str, Any]:
        return self._request("GET", "/account")

    def positions(self) -> List[Dict[str, Any]]:
        return self._request("GET", "/positions") or []

    def submit_market_order(self, symbol: str, qty: float, side: str, time_in_force: str = "day") -> Dict[str, Any]:
        if side not in ("buy", "sell"):
            raise ValueError("side must be buy or sell")
        return self._request("POST", "/orders", json={"symbol": symbol, "qty": str(qty), "side": side,
                                                      "type": "market", "time_in_force": time_in_force})

    def cancel_all_orders(self) -> None:
        self._request("DELETE", "/orders")
