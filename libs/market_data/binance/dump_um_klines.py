from __future__ import annotations

import datetime as dt
from pathlib import Path

from binance_historical_data import BinanceDataDumper
from libs.market_data.binance.count_local_kline_csvs import count_local_kline_csvs
from libs.network.ensure_host_resolves import ensure_host_resolves


def dump_um_klines(
    *,
    root_dir: str | Path,
    ticker: str,
    interval: str,
    date_start: dt.date,
    date_end: dt.date,
    update_existing: bool,
) -> Path:
    root = Path(root_dir)
    root.mkdir(parents=True, exist_ok=True)

    ensure_host_resolves("data.binance.vision")

    before_count = count_local_kline_csvs(root_dir=root, ticker=ticker, interval=interval)

    data_dumper = BinanceDataDumper(
        path_dir_where_to_dump=str(root),
        asset_class="um",
        data_type="klines",
        data_frequency=interval,
    )

    data_dumper.get_list_all_trading_pairs = lambda: [ticker]

    data_dumper.dump_data(
        tickers=[ticker],
        date_start=date_start,
        date_end=date_end,
        is_to_update_existing=update_existing,
        tickers_to_exclude=[],
    )

    after_count = count_local_kline_csvs(root_dir=root, ticker=ticker, interval=interval)
    if after_count <= before_count:
        print(f"⚠️  Aucun nouveau fichier téléchargé. Fichiers existants: {before_count}")
        # Ne pas lever d'erreur, car les données peuvent déjà être complètes
        print("ℹ️  Les données existantes sont probablement complètes")
        return root

    return root
