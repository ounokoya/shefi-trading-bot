from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from libs.backtest_triple_macd_simple_alignment.config import load_config_yaml  # noqa: E402
from libs.backtest_triple_macd_simple_alignment.engine import (  # noqa: E402
    BacktestTripleMacdSimpleAlignmentConfig,
    build_agent_config,
    run_backtest_triple_macd_simple_alignment,
)
from libs.blocks.add_macd_tv_columns_df import add_macd_tv_columns_df  # noqa: E402


def _interval_to_bybit(interval: str) -> str:
    s = str(interval).strip()
    if not s:
        raise ValueError("interval cannot be empty")

    if s.isdigit():
        return s

    s_lower = s.lower()
    if s_lower in {"d", "1d"}:
        return "D"
    if s_lower in {"w", "1w"}:
        return "W"
    if s == "M" or s_lower in {"1mo", "mo", "1month"}:
        return "M"

    import re

    m = re.fullmatch(r"(\d+)([mhd])", s_lower)
    if not m:
        raise ValueError(f"unsupported interval format: {interval}")

    n = int(m.group(1))
    unit = m.group(2)
    if unit == "m":
        minutes = n
    elif unit == "h":
        minutes = n * 60
    elif unit == "d":
        if n != 1:
            raise ValueError(f"Bybit only supports daily interval as 1d/D, got: {interval}")
        return "D"
    else:
        raise ValueError(f"unsupported interval unit: {unit}")

    allowed_minutes = {1, 3, 5, 15, 30, 60, 120, 240, 360, 720}
    if minutes not in allowed_minutes:
        allowed_str = ", ".join(str(x) for x in sorted(allowed_minutes))
        raise ValueError(
            f"unsupported minute interval for Bybit: {minutes} (from {interval}). Allowed: {allowed_str}"
        )
    return str(minutes)


def _fetch_bybit_klines_range(
    *,
    symbol: str,
    interval: str,
    start_ms: int,
    end_ms: int,
    page_limit: int,
    category: str,
    base_url: str,
    timeout_s: float,
    max_pages: int = 250,
) -> pd.DataFrame:
    import requests

    url = f"{base_url.rstrip('/')}/v5/market/kline"

    if int(page_limit) <= 0:
        page_limit = 200
    if int(page_limit) > 1000:
        page_limit = 1000

    start = int(start_ms)
    cur_end = int(end_ms)
    if int(cur_end) < int(start):
        return pd.DataFrame([])

    seen: set[int] = set()
    out_rows: list[dict[str, object]] = []

    for _ in range(int(max_pages)):
        params: dict[str, Any] = {
            "category": str(category),
            "symbol": str(symbol),
            "interval": str(interval),
            "limit": str(int(page_limit)),
            "start": str(int(start)),
            "end": str(int(cur_end)),
        }
        r = requests.get(url, params=params, timeout=float(timeout_s))
        r.raise_for_status()
        payload = r.json()
        if str(payload.get("retCode")) != "0":
            raise RuntimeError(f"Bybit retCode={payload.get('retCode')} retMsg={payload.get('retMsg')}")

        result = payload.get("result") or {}
        rows = result.get("list") or []
        if not rows:
            break

        ts_list: list[int] = []
        for row in rows:
            if not isinstance(row, list) or len(row) < 1:
                continue
            try:
                ts_list.append(int(row[0]))
            except Exception:
                continue
        if not ts_list:
            break

        oldest = int(min(ts_list))
        for row in rows:
            if not isinstance(row, list) or len(row) < 6:
                continue
            ts = int(row[0])
            if ts in seen:
                continue
            seen.add(ts)
            if int(start_ms) <= ts <= int(end_ms):
                out_rows.append(
                    {
                        "ts": ts,
                        "open": float(row[1]),
                        "high": float(row[2]),
                        "low": float(row[3]),
                        "close": float(row[4]),
                        "volume": float(row[5]),
                    }
                )

        if int(oldest) <= int(start_ms):
            break
        nxt_end = int(oldest) - 1
        if int(nxt_end) >= int(cur_end):
            break
        cur_end = int(nxt_end)

    df = pd.DataFrame(out_rows)
    if len(df):
        df = df.sort_values("ts").reset_index(drop=True)
        df["dt"] = pd.to_datetime(df["ts"], unit="ms", utc=True).dt.strftime("%Y-%m-%d %H:%M:%S UTC")
    return df


def _add_dt_cols(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    df2 = df.copy()
    for c in cols:
        if c not in df2.columns:
            continue
        s = pd.to_numeric(df2[c], errors="coerce")
        df2[f"{c}_dt"] = pd.to_datetime(s, unit="ms", utc=True).dt.strftime("%Y-%m-%d %H:%M:%S UTC")
    return df2


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True, help="Path to YAML config")
    ap.add_argument("--out-dir", default=None)
    args = ap.parse_args()

    cfg = load_config_yaml(str(args.config))

    start_ts = int(pd.Timestamp(cfg.bybit.start, tz="UTC").value // 1_000_000)
    end_ts = int((pd.Timestamp(cfg.bybit.end, tz="UTC") + pd.Timedelta(days=1) - pd.Timedelta(milliseconds=1)).value // 1_000_000)

    warmup_bars = int(cfg.bybit.warmup_bars)
    if int(warmup_bars) <= 0:
        warmup_bars = max(int(cfg.indicators.macd_slow.slow) * 3, 50)

    def _interval_to_minutes(interval: str) -> int:
        s = str(interval).strip()
        if not s:
            return 0
        if s.isdigit():
            return int(s)
        s_lower = s.lower()
        if s_lower.endswith("m") and s_lower[:-1].isdigit():
            return int(s_lower[:-1])
        if s_lower.endswith("h") and s_lower[:-1].isdigit():
            return int(s_lower[:-1]) * 60
        if s_lower in {"d", "1d"}:
            return 24 * 60
        return 0

    mins = _interval_to_minutes(str(cfg.bybit.interval))
    warmup_ms = int(warmup_bars) * int(mins) * 60_000 if int(mins) > 0 else 0
    fetch_start_ms = int(max(0, int(start_ts) - int(warmup_ms)))

    df = _fetch_bybit_klines_range(
        symbol=str(cfg.bybit.symbol),
        interval=_interval_to_bybit(str(cfg.bybit.interval)),
        start_ms=int(fetch_start_ms),
        end_ms=int(end_ts),
        page_limit=int(cfg.bybit.limit),
        category=str(cfg.bybit.category),
        base_url=str(cfg.bybit.base_url),
        timeout_s=30.0,
    )

    if not len(df):
        raise RuntimeError("no klines fetched")

    df = add_macd_tv_columns_df(
        df,
        close_col="close",
        fast_period=int(cfg.indicators.macd_slow.fast),
        slow_period=int(cfg.indicators.macd_slow.slow),
        signal_period=int(cfg.indicators.macd_slow.signal),
        out_line_col="macd_line_slow",
        out_signal_col="macd_signal_slow",
        out_hist_col="macd_hist_slow",
    )
    df = add_macd_tv_columns_df(
        df,
        close_col="close",
        fast_period=int(cfg.indicators.macd_medium.fast),
        slow_period=int(cfg.indicators.macd_medium.slow),
        signal_period=int(cfg.indicators.macd_medium.signal),
        out_line_col="macd_line_medium",
        out_signal_col="macd_signal_medium",
        out_hist_col="macd_hist_medium",
    )
    df = add_macd_tv_columns_df(
        df,
        close_col="close",
        fast_period=int(cfg.indicators.macd_fast.fast),
        slow_period=int(cfg.indicators.macd_fast.slow),
        signal_period=int(cfg.indicators.macd_fast.signal),
        out_line_col="macd_line_fast",
        out_signal_col="macd_signal_fast",
        out_hist_col="macd_hist_fast",
    )

    agent_cfg = build_agent_config(
        ts_col="ts",
        dt_col="dt",
        close_col="close",
        hist_zero_policy=str(cfg.agent.hist_zero_policy),
        require_hists_rising_on_entry=bool(cfg.agent.require_hists_rising_on_entry),
        slow_role=cfg.agent.slow,
        medium_role=cfg.agent.medium,
        fast_role=cfg.agent.fast,
    )

    bt_cfg = BacktestTripleMacdSimpleAlignmentConfig(
        ts_col="ts",
        open_col="open",
        high_col="high",
        low_col="low",
        close_col="close",
        fee_rate=float(cfg.backtest.fee_rate),
        use_net=bool(cfg.backtest.use_net),
        exit_mode=str(cfg.backtest.exit_mode),
        tp_pct=float(cfg.backtest.tp_pct),
        trailing_stop_pct=float(cfg.backtest.trailing_stop_pct),
        sl_pct=float(cfg.backtest.sl_pct),
        entry_on_next_bar=bool(cfg.backtest.entry_on_next_bar),
    )

    res = run_backtest_triple_macd_simple_alignment(
        df=df,
        agent_cfg=agent_cfg,
        bt_cfg=bt_cfg,
        start_ts=int(start_ts),
        end_ts=int(end_ts),
        max_signals=int(cfg.backtest.max_signals),
    )

    trades = res["trades"]
    equity = res["equity"]
    summary = res["summary"]

    out_dir = Path(str(args.out_dir) if args.out_dir is not None else cfg.output.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    trades = _add_dt_cols(trades, ["entry_ts", "exit_ts", "entry_signal_ts"])
    equity = _add_dt_cols(equity, ["ts"])

    if bool(cfg.output.save_csv):
        trades_csv = out_dir / "trades.csv"
        equity_csv = out_dir / "equity.csv"
        trades.to_csv(trades_csv, index=False, float_format="%.6f")
        equity.to_csv(equity_csv, index=False, float_format="%.6f")
        print(f"Wrote: {trades_csv}")
        print(f"Wrote: {equity_csv}")

    print("Backtest summary:")
    print(f"- n_trades: {summary.get('n_trades')}")
    print(f"- winrate: {float(summary.get('winrate') or 0.0):.4f}")
    print(f"- equity_end: {float(summary.get('equity_end') or 0.0):.6f}")
    print(f"- max_dd: {float(summary.get('max_dd') or 0.0):.6f}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
