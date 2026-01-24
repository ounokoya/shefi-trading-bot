from __future__ import annotations

from pathlib import Path


def count_local_kline_csvs(*, root_dir: str | Path, ticker: str, interval: str) -> int:
    root = Path(root_dir)
    count = 0
    for p in root.rglob("*.csv"):
        if "CHECKSUM" in p.name:
            continue
        ps = p.as_posix()
        if ticker not in ps:
            continue
        if f"/{interval}/" not in ps and f"_{interval}_" not in p.name and interval not in ps:
            continue
        count += 1
    return count
