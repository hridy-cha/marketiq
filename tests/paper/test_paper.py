from datetime import date
from types import SimpleNamespace

import pytest

from marketiq.paper.broker import PaperBroker, PaperTradingError


def test_create_buy_sell_and_portfolio(seeded_session):
    PaperBroker.create_account(seeded_session, "t", 10_000.0)
    b = PaperBroker(seeded_session, "t", commission_bps=0, slippage_bps=0)
    px = b.last_price("AAPL", date(2024, 6, 28))
    fill = b.buy("AAPL", notional=5_000, trade_date=date(2024, 6, 28), reason="test")
    assert fill.price == pytest.approx(px) and fill.quantity == pytest.approx(5_000 / px)
    pf = b.portfolio(date(2024, 6, 28))
    assert pf["cash"] == pytest.approx(5_000) and pf["equity"] == pytest.approx(10_000)
    assert pf["positions"][0]["symbol"] == "AAPL"
    later = b.portfolio(date(2024, 12, 31))
    assert later["equity"] != pytest.approx(10_000)  # marked at the later real close
    sell = b.sell("AAPL", trade_date=date(2024, 12, 31), reason="done")
    assert sell.quantity == pytest.approx(fill.quantity)
    assert b.positions() == []
    assert b.portfolio()["equity"] == pytest.approx(b.account.cash)
    assert [t.side for t in b.trades()] == ["SELL", "BUY"]




def test_errors(seeded_session):
    PaperBroker.create_account(seeded_session, "e", 100.0)
    b = PaperBroker(seeded_session, "e")
    with pytest.raises(PaperTradingError):
        b.buy("AAPL", notional=1_000, trade_date=date(2024, 6, 28))
    with pytest.raises(PaperTradingError):
        b.sell("MSFT", trade_date=date(2024, 6, 28))
    with pytest.raises(PaperTradingError):
        b.buy("ZZZZ", notional=10, trade_date=date(2024, 6, 28))


def test_apply_signals_enters_and_exits(seeded_session):
    PaperBroker.create_account(seeded_session, "s", 20_000.0)
    b = PaperBroker(seeded_session, "s", commission_bps=0, slippage_bps=0)
    row = lambda sym, sig, sc: SimpleNamespace(symbol=sym, score=SimpleNamespace(score=sc), signal=SimpleNamespace(signal=sig, rule="r"))
    fills = b.apply_signals([row("AAPL", "BUY", 80), row("MSFT", "BUY", 70)], max_positions=2, trade_date=date(2024, 6, 28))
    assert sorted(f.symbol for f in fills) == ["AAPL", "MSFT"]
    assert all(abs(f.quantity * f.price - 10_000) < 1 for f in fills)
    fills2 = b.apply_signals([row("AAPL", "SELL", 20), row("MSFT", "HOLD", 40)], max_positions=2, exit_score=45, trade_date=date(2024, 7, 31))
    assert sorted((f.symbol, f.side) for f in fills2) == [("AAPL", "SELL"), ("MSFT", "SELL")]
    assert "below exit" in next(f.reason for f in fills2 if f.symbol == "MSFT")
    hist = b.equity_history()
    assert [h.as_of for h in hist] == [date(2024, 6, 28), date(2024, 7, 31)]


def test_reset_account(seeded_session):
    PaperBroker.create_account(seeded_session, "r", 1_000.0)
    b = PaperBroker(seeded_session, "r", commission_bps=0, slippage_bps=0)
    b.buy("AAPL", notional=500, trade_date=date(2024, 6, 28))
    acct = PaperBroker.reset_account(seeded_session, "r", 2_000.0)
    assert acct.cash == 2_000.0 and acct.positions == [] and acct.trades == []


def test_alpaca_adapter_refuses_live_without_flag(monkeypatch):
    from marketiq.config.settings import get_settings
    from marketiq.paper.alpaca_adapter import AlpacaAdapter, LiveTradingDisabled

    monkeypatch.setenv("ENABLE_LIVE_TRADING", "false")
    get_settings.cache_clear()
    with pytest.raises(LiveTradingDisabled):
        AlpacaAdapter(api_key="k", api_secret="s", base_url="https://api.alpaca.markets")
    a = AlpacaAdapter(api_key="k", api_secret="s", base_url="https://paper-api.alpaca.markets")
    assert a.is_paper
    get_settings.cache_clear()
