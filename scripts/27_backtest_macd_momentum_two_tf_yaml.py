from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from libs.agents.macd_momentum_two_tf_cycle_agent import MacdMomentumTwoTFConfig
from libs.backtest_macd_momentum_two_tf.config import load_config_yaml
from libs.backtest_macd_momentum_two_tf.engine import BacktestMacdMomentumTwoTFConfig, run_backtest_macd_momentum_two_tf
from libs.blocks.add_macd_tv_columns_df import add_macd_tv_columns_df
from libs.indicators.momentum.cci_tv import cci_tv


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
        params = {
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


def _add_ccis(df: pd.DataFrame, *, cci_fast: int, cci_medium: int, cci_slow: int) -> tuple[pd.DataFrame, str, str, str]:
    out = df.copy()
    high = pd.to_numeric(out["high"], errors="coerce").astype(float).tolist()
    low = pd.to_numeric(out["low"], errors="coerce").astype(float).tolist()
    close = pd.to_numeric(out["close"], errors="coerce").astype(float).tolist()

    cci_fast_col = f"cci_{int(cci_fast)}"
    cci_medium_col = f"cci_{int(cci_medium)}"
    cci_slow_col = f"cci_{int(cci_slow)}"

    out[cci_fast_col] = cci_tv(high, low, close, int(cci_fast))
    out[cci_medium_col] = cci_tv(high, low, close, int(cci_medium))
    out[cci_slow_col] = cci_tv(high, low, close, int(cci_slow))

    return out, cci_fast_col, cci_medium_col, cci_slow_col


def _add_stoch(df: pd.DataFrame, *, k_period: int, d_period: int) -> pd.DataFrame:
    import numpy as np

    out = df.copy()
    k_period2 = int(k_period)
    d_period2 = int(d_period)
    if int(k_period2) < 1:
        raise ValueError("stoch.k must be >= 1")
    if int(d_period2) < 1:
        raise ValueError("stoch.d must be >= 1")

    low_s = pd.to_numeric(out["low"], errors="coerce").astype(float)
    high_s = pd.to_numeric(out["high"], errors="coerce").astype(float)
    close_s = pd.to_numeric(out["close"], errors="coerce").astype(float)

    ll = low_s.rolling(window=k_period2, min_periods=k_period2).min()
    hh = high_s.rolling(window=k_period2, min_periods=k_period2).max()
    denom = (hh - ll).astype(float)
    numer = (close_s - ll).astype(float)

    k = 100.0 * (numer / denom.replace(0.0, np.nan))
    out["stoch_k"] = k
    out["stoch_d"] = out["stoch_k"].rolling(window=d_period2, min_periods=d_period2).mean()
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True, help="Path to YAML config")
    ap.add_argument("--out-dir", default=None)
    args = ap.parse_args()

    cfg = load_config_yaml(str(args.config))

    start_ts = int(pd.Timestamp(cfg.bybit.start, tz="UTC").value // 1_000_000)
    end_ts = int(
        (pd.Timestamp(cfg.bybit.end, tz="UTC") + pd.Timedelta(days=1) - pd.Timedelta(milliseconds=1)).value // 1_000_000
    )

    warmup_bars = int(cfg.bybit.warmup_bars)
    if int(warmup_bars) <= 0:
        warmup_bars = max(
            int(cfg.indicators.exec_cci.slow),
            int(cfg.indicators.ctx_cci.slow),
            int(cfg.indicators.exec_macd.slow) * 3,
            int(cfg.indicators.ctx_macd.slow) * 3,
            50,
        )

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

    exec_min = _interval_to_minutes(str(cfg.bybit.exec_interval))
    ctx_min = _interval_to_minutes(str(cfg.bybit.ctx_interval))
    exec_warmup_ms = int(warmup_bars) * int(exec_min) * 60_000 if int(exec_min) > 0 else 0
    ctx_warmup_ms = int(warmup_bars) * int(ctx_min) * 60_000 if int(ctx_min) > 0 else 0
    exec_fetch_start_ms = int(max(0, int(start_ts) - int(exec_warmup_ms)))
    ctx_fetch_start_ms = int(max(0, int(start_ts) - int(ctx_warmup_ms)))

    df_exec = _fetch_bybit_klines_range(
        symbol=str(cfg.bybit.symbol),
        interval=_interval_to_bybit(str(cfg.bybit.exec_interval)),
        start_ms=int(exec_fetch_start_ms),
        end_ms=int(end_ts),
        page_limit=int(cfg.bybit.exec_limit),
        category=str(cfg.bybit.category),
        base_url=str(cfg.bybit.base_url),
        timeout_s=30.0,
    )
    df_ctx = _fetch_bybit_klines_range(
        symbol=str(cfg.bybit.symbol),
        interval=_interval_to_bybit(str(cfg.bybit.ctx_interval)),
        start_ms=int(ctx_fetch_start_ms),
        end_ms=int(end_ts),
        page_limit=int(cfg.bybit.ctx_limit),
        category=str(cfg.bybit.category),
        base_url=str(cfg.bybit.base_url),
        timeout_s=30.0,
    )

    if not len(df_exec):
        raise RuntimeError("no exec klines fetched")
    if not len(df_ctx):
        raise RuntimeError("no ctx klines fetched")

    df_exec = add_macd_tv_columns_df(
        df_exec,
        close_col="close",
        fast_period=int(cfg.indicators.exec_macd.fast),
        slow_period=int(cfg.indicators.exec_macd.slow),
        signal_period=int(cfg.indicators.exec_macd.signal),
    )
    df_ctx = add_macd_tv_columns_df(
        df_ctx,
        close_col="close",
        fast_period=int(cfg.indicators.ctx_macd.fast),
        slow_period=int(cfg.indicators.ctx_macd.slow),
        signal_period=int(cfg.indicators.ctx_macd.signal),
    )

    df_exec, cci_exec_fast_col, cci_exec_medium_col, cci_exec_slow_col = _add_ccis(
        df_exec,
        cci_fast=int(cfg.indicators.exec_cci.fast),
        cci_medium=int(cfg.indicators.exec_cci.medium),
        cci_slow=int(cfg.indicators.exec_cci.slow),
    )
    df_ctx, cci_ctx_fast_col, cci_ctx_medium_col, cci_ctx_slow_col = _add_ccis(
        df_ctx,
        cci_fast=int(cfg.indicators.ctx_cci.fast),
        cci_medium=int(cfg.indicators.ctx_cci.medium),
        cci_slow=int(cfg.indicators.ctx_cci.slow),
    )

    df_exec = _add_stoch(df_exec, k_period=int(cfg.indicators.stoch.k), d_period=int(cfg.indicators.stoch.d))

    agent_cfg = MacdMomentumTwoTFConfig(
        ts_col="ts",
        dt_col="dt",
        open_col="open",
        high_col="high",
        low_col="low",
        close_col="close",
        hist_col="macd_hist",
        cci_exec_fast_col=str(cci_exec_fast_col),
        cci_exec_medium_col=str(cci_exec_medium_col),
        cci_exec_slow_col=str(cci_exec_slow_col),
        cci_exec_fast_period=int(cfg.indicators.exec_cci.fast),
        cci_exec_medium_period=int(cfg.indicators.exec_cci.medium),
        cci_exec_slow_period=int(cfg.indicators.exec_cci.slow),
        cci_ctx_fast_col=str(cci_ctx_fast_col),
        cci_ctx_medium_col=str(cci_ctx_medium_col),
        cci_ctx_slow_col=str(cci_ctx_slow_col),
        cci_ctx_fast_period=int(cfg.indicators.ctx_cci.fast),
        cci_ctx_medium_period=int(cfg.indicators.ctx_cci.medium),
        cci_ctx_slow_period=int(cfg.indicators.ctx_cci.slow),
        min_abs_force_exec=float(cfg.agent.min_abs_force_exec),
        min_abs_force_ctx=float(cfg.agent.min_abs_force_ctx),
        cci_global_extreme_level_exec=float(cfg.agent.exec_cci_extreme),
        cci_global_extreme_level_ctx=float(cfg.agent.ctx_cci_extreme),
        take_exec_cci_extreme_if_ctx_not_extreme=bool(cfg.agent.take_exec_cci_extreme_if_ctx_not_extreme),
        take_exec_and_ctx_cci_extreme=bool(cfg.agent.take_exec_and_ctx_cci_extreme),
        signal_on_ctx_flip_if_exec_aligned=bool(cfg.agent.signal_on_ctx_flip_if_exec_aligned),
    )

    bt_cfg = BacktestMacdMomentumTwoTFConfig(
        fee_rate=float(cfg.backtest.fee_rate),
        exit_mode=str(cfg.backtest.exit_mode),
        tp_pct=float(cfg.backtest.tp_pct),
        trailing_stop_pct=float(cfg.backtest.trailing_stop_pct),
        sl_pct=float(cfg.backtest.sl_pct),
        stoch_high=float(cfg.backtest.stoch_high),
        stoch_low=float(cfg.backtest.stoch_low),
        stoch_wait_extreme=bool(cfg.backtest.stoch_wait_extreme),
    )

    res = run_backtest_macd_momentum_two_tf(
        df_exec=df_exec,
        df_ctx=df_ctx,
        agent_cfg=agent_cfg,
        bt_cfg=bt_cfg,
        start_ts=start_ts,
        end_ts=end_ts,
        max_signals=int(cfg.backtest.max_signals),
    )

    out_dir = Path(str(args.out_dir) if args.out_dir is not None else cfg.output.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    trades = res["trades"]
    equity = res["equity"]

    if bool(cfg.output.save_csv):
        trades.to_csv(out_dir / "trades.csv", index=False, float_format="%.6f")
        equity.to_csv(out_dir / "equity.csv", index=False, float_format="%.6f")

    print("# Backtest MacdMomentumTwoTF (YAML)")
    print(f"symbol={cfg.bybit.symbol} exec={cfg.bybit.exec_interval} ctx={cfg.bybit.ctx_interval} start={cfg.bybit.start} end={cfg.bybit.end}")
    print(f"exit_mode={cfg.backtest.exit_mode} fee_rate={cfg.backtest.fee_rate}")

    print("# Summary")
    for k, v in res["summary"].items():
        print(f"{k}: {v}")

    n_top = int(cfg.output.print_top_reasons)

    def _print_top(df: pd.DataFrame, *, title: str) -> None:
        if int(n_top) <= 0:
            return
        if df is None or (not len(df)):
            return

        if int(len(df)) <= int(n_top):
            print(f"# {title} (all {len(df)} groups)")
            print(df.to_string(index=False))
            return

        print(f"# {title} (top {n_top} by pnl_sum)")
        print(df.head(n_top).to_string(index=False))
        print(f"# {title} (bottom {n_top} by pnl_sum)")
        print(df.tail(n_top).to_string(index=False))

    _print_top(res.get("by_entry_reason"), title="By entry_reason")
    _print_top(res.get("by_entry_group"), title="By entry_reason|signal_kind")
    _print_top(res.get("by_exit_reason"), title="By exit_reason")

    print(f"Wrote: {out_dir / 'trades.csv'}")
    print(f"Wrote: {out_dir / 'equity.csv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
