"""
Simulated paper broker backed by the database.

Market orders fill immediately at the latest stored close for the symbol (or at a price the caller
supplies), with configurable commission and slippage. No money moves anywhere. The broker also
offers `apply_signals`, which rebalances the account from the latest scan the same way the
backtester does, so paper trading and backtesting share one set of rules.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import date
from typing import Dict, List, Optional

from sqlalchemy import select
from sqlalchemy.orm import Session

from marketiq.data import repository as repo
from marketiq.data.models import PaperAccount, PaperEquityPoint, PaperPosition, PaperTrade

logger = logging.getLogger(__name__)


class PaperTradingError(RuntimeError):
    pass


@dataclass
class Fill:
    symbol: str
    side: str
    quantity: float
    price: float
    commission: float
    trade_date: date
    reason: str


class PaperBroker:
    def __init__(self, session: Session, account_name: str = "default", commission_bps: float = 5.0, slippage_bps: float = 5.0):
        self.session = session
        self.commission_bps = commission_bps
        self.slippage_bps = slippage_bps
        self.account = session.scalar(select(PaperAccount).where(PaperAccount.name == account_name))
        if self.account is None:
            raise PaperTradingError(f"Paper account '{account_name}' does not exist; create it first")

    # ------------------------------------------------------------------
    @classmethod
    def create_account(cls, session: Session, name: str = "default", starting_cash: float = 100_000.0) -> PaperAccount:
        existing = session.scalar(select(PaperAccount).where(PaperAccount.name == name))
        if existing:
            return existing
        acct = PaperAccount(name=name, starting_cash=starting_cash, cash=starting_cash)
        session.add(acct)
        session.flush()
        return acct

    @classmethod
    def reset_account(cls, session: Session, name: str = "default", starting_cash: Optional[float] = None) -> PaperAccount:
        acct = session.scalar(select(PaperAccount).where(PaperAccount.name == name))
        if acct is None:
            return cls.create_account(session, name, starting_cash or 100_000.0)
        acct.positions.clear()
        acct.trades.clear()
        for e in session.scalars(select(PaperEquityPoint).where(PaperEquityPoint.account_id == acct.id)):
            session.delete(e)
        acct.starting_cash = starting_cash or acct.starting_cash
        acct.cash = acct.starting_cash
        session.flush()
        return acct

    # ------------------------------------------------------------------
    def last_price(self, symbol: str, as_of: Optional[date] = None) -> Optional[float]:
        df = repo.get_price_frame(self.session, symbol, end=as_of)
        if df.empty:
            return None
        return float(df["close"].iloc[-1])

    def positions(self) -> List[PaperPosition]:
        return list(self.account.positions)

    def portfolio(self, as_of: Optional[date] = None) -> dict:
        rows = []
        invested = 0.0
        for p in self.positions():
            px = self.last_price(p.symbol, as_of) or p.avg_cost
            value = p.quantity * px
            invested += value
            rows.append({"symbol": p.symbol, "quantity": p.quantity, "avg_cost": p.avg_cost, "last_price": px,
                         "market_value": value, "unrealized_pnl": (px - p.avg_cost) * p.quantity,
                         "unrealized_pnl_pct": (px / p.avg_cost - 1.0) if p.avg_cost else 0.0})
        equity = self.account.cash + invested
        return {"account": self.account.name, "cash": self.account.cash, "invested": invested, "equity": equity,
                "starting_cash": self.account.starting_cash, "total_return": equity / self.account.starting_cash - 1.0,
                "positions": rows}

    def trades(self, limit: int = 200) -> List[PaperTrade]:
        return list(self.session.scalars(
            select(PaperTrade).where(PaperTrade.account_id == self.account.id).order_by(PaperTrade.created_at.desc()).limit(limit)))

    def equity_history(self) -> List[PaperEquityPoint]:
        return list(self.session.scalars(
            select(PaperEquityPoint).where(PaperEquityPoint.account_id == self.account.id).order_by(PaperEquityPoint.as_of)))

    def mark(self, as_of: Optional[date] = None) -> PaperEquityPoint:
        pf = self.portfolio(as_of)
        d = as_of or date.today()
        point = self.session.scalar(select(PaperEquityPoint).where(PaperEquityPoint.account_id == self.account.id, PaperEquityPoint.as_of == d))
        if point is None:
            point = PaperEquityPoint(account_id=self.account.id, as_of=d, equity=pf["equity"], cash=pf["cash"])
            self.session.add(point)
        else:
            point.equity, point.cash = pf["equity"], pf["cash"]
        self.session.flush()
        return point

    # ------------------------------------------------------------------
    def buy(self, symbol: str, notional: Optional[float] = None, quantity: Optional[float] = None,
            price: Optional[float] = None, trade_date: Optional[date] = None, reason: str = "") -> Fill:
        px = price or self.last_price(symbol, trade_date)
        if px is None:
            raise PaperTradingError(f"No price available for {symbol}")
        fill_px = px * (1 + self.slippage_bps / 1e4)
        if quantity is None:
            if notional is None or notional <= 0:
                raise PaperTradingError("Provide notional or quantity")
            quantity = notional / fill_px
        cost = quantity * fill_px
        commission = cost * self.commission_bps / 1e4
        if cost + commission > self.account.cash + 1e-9:
            raise PaperTradingError(f"Insufficient cash: need {cost + commission:.2f}, have {self.account.cash:.2f}")
        self.account.cash -= cost + commission
        pos = next((p for p in self.account.positions if p.symbol == symbol), None)
        if pos is None:
            pos = PaperPosition(account_id=self.account.id, symbol=symbol, quantity=0.0, avg_cost=0.0)
            self.account.positions.append(pos)
        total_cost = pos.avg_cost * pos.quantity + fill_px * quantity
        pos.quantity += quantity
        pos.avg_cost = total_cost / pos.quantity
        d = trade_date or date.today()
        self.session.add(PaperTrade(account_id=self.account.id, symbol=symbol, side="BUY", quantity=quantity,
                                    price=fill_px, commission=commission, trade_date=d, reason=reason))
        self.session.flush()
        return Fill(symbol, "BUY", quantity, fill_px, commission, d, reason)

    def sell(self, symbol: str, quantity: Optional[float] = None, price: Optional[float] = None,
             trade_date: Optional[date] = None, reason: str = "") -> Fill:
        pos = next((p for p in self.account.positions if p.symbol == symbol), None)
        if pos is None or pos.quantity <= 0:
            raise PaperTradingError(f"No position in {symbol}")
        quantity = pos.quantity if quantity is None else min(quantity, pos.quantity)
        px = price or self.last_price(symbol, trade_date)
        if px is None:
            raise PaperTradingError(f"No price available for {symbol}")
        fill_px = px * (1 - self.slippage_bps / 1e4)
        proceeds = quantity * fill_px
        commission = proceeds * self.commission_bps / 1e4
        self.account.cash += proceeds - commission
        pos.quantity -= quantity
        if pos.quantity <= 1e-9:
            self.account.positions.remove(pos)
            self.session.delete(pos)
        d = trade_date or date.today()
        self.session.add(PaperTrade(account_id=self.account.id, symbol=symbol, side="SELL", quantity=quantity,
                                    price=fill_px, commission=commission, trade_date=d, reason=reason))
        self.session.flush()
        return Fill(symbol, "SELL", quantity, fill_px, commission, d, reason)

    # ------------------------------------------------------------------
    def apply_signals(self, rows, max_positions: int = 8, exit_score: float = 45.0, enter_on=("BUY",),
                      trade_date: Optional[date] = None) -> List[Fill]:
        """Rebalance from scan rows using the backtester's rules: exit on SELL or low score, fill free
        slots with the best entry signals, equal notional per slot."""
        fills: List[Fill] = []
        by_sym = {r.symbol: r for r in rows}
        for pos in list(self.positions()):
            r = by_sym.get(pos.symbol)
            if r is None:
                continue
            if r.signal.signal == "SELL":
                fills.append(self.sell(pos.symbol, trade_date=trade_date, reason=f"SELL signal ({r.signal.rule})"))
            elif r.score.score < exit_score:
                fills.append(self.sell(pos.symbol, trade_date=trade_date, reason=f"score {r.score.score:.1f} below exit {exit_score:.0f}"))
        held = {p.symbol for p in self.positions()}
        free = max_positions - len(held)
        candidates = sorted([r for r in rows if r.signal.signal in enter_on and r.symbol not in held],
                            key=lambda r: r.score.score, reverse=True)[:max(0, free)]
        if candidates:
            equity = self.portfolio(trade_date)["equity"]
            cash_after_fees = self.account.cash / (1 + self.commission_bps / 1e4)
            slot = min(equity / max_positions, cash_after_fees / len(candidates))
            for r in candidates:
                try:
                    fills.append(self.buy(r.symbol, notional=slot, trade_date=trade_date,
                                          reason=f"{r.signal.signal} score {r.score.score:.1f} ({r.signal.rule})"))
                except PaperTradingError as exc:
                    logger.warning("paper buy skipped for %s: %s", r.symbol, exc)
        self.mark(trade_date)
        return fills
