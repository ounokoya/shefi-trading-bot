import pandas as pd
import requests
import time
import logging
from pathlib import Path
from typing import Optional

def _interval_to_bybit(interval: str) -> str:
    # Simplified mapping
    it = str(interval or "").strip().lower()
    if it.endswith("m") and it[:-1].isdigit():
        return str(int(it[:-1]))
    if it.endswith("h") and it[:-1].isdigit():
        return str(int(it[:-1]) * 60)
    if it.endswith("min") and it[:-3].isdigit():
        return str(int(it[:-3]))
    if it == "1d":
        return "D"
    return it

def _fetch_bybit_klines(
    *,
    symbol: str,
    interval: str,
    limit: int,
    category: str,
    base_url: str,
    timeout_s: float,
    start_ms: Optional[int] = None,
) -> list:
    url = f"{base_url.rstrip('/')}/v5/market/kline"
    params = {
        "category": category,
        "symbol": symbol,
        "interval": interval,
        "limit": str(int(limit)),
    }
    if start_ms is not None:
        params["start"] = str(int(start_ms))
    
    try:
        r = requests.get(url, params=params, timeout=timeout_s)
        r.raise_for_status()
        payload = r.json()
    except Exception as e:
        logging.error(f"Request failed: {e}")
        return []

    if str(payload.get("retCode")) != "0":
        logging.error(f"Bybit retCode={payload.get('retCode')} retMsg={payload.get('retMsg')}")
        return []

    result = payload.get("result") or {}
    rows = result.get("list") or []
    out = []
    for row in rows:
        if not isinstance(row, list) or len(row) < 5:
            continue
        out.append({
            'ts': int(row[0]),
            'open': float(row[1]),
            'high': float(row[2]),
            'low': float(row[3]),
            'close': float(row[4]),
            'volume': float(row[5])
        })
    
    out.sort(key=lambda x: x['ts'])
    return out

def fetch_bybit_klines_range(
    symbol: str,
    interval: str,
    start_ms: int,
    end_ms: int,
    base_url: str = "https://api.bybit.com",
    category: str = "linear"
) -> pd.DataFrame:
    bybit_interval = _interval_to_bybit(interval)
    page_limit = 1000
    current_start = start_ms
    all_rows = []
    
    logging.info(f"Fetching {symbol} {interval} from {start_ms} to {end_ms}")
    
    while True:
        rows = _fetch_bybit_klines(
            symbol=symbol,
            interval=bybit_interval,
            limit=page_limit,
            category=category,
            base_url=base_url,
            timeout_s=30,
            start_ms=current_start,
        )
        
        if not rows:
            break
            
        first_ts = rows[0]['ts']
        last_ts = rows[-1]['ts']
        
        # Filter valid rows
        new_rows = [r for r in rows if r['ts'] >= current_start and r['ts'] <= end_ms]
        
        if not new_rows:
            if first_ts > end_ms:
                break
            if last_ts < current_start:
                break
            # Gap handling
            current_start += int(bybit_interval) * 60 * 1000 * page_limit
            continue

        # Add unique rows
        seen_ts = {r['ts'] for r in all_rows}
        cnt = 0
        for r in new_rows:
            if r['ts'] not in seen_ts:
                all_rows.append(r)
                cnt += 1
        
        logging.info(f"Fetched {len(rows)} rows [{first_ts}..{last_ts}]. Added {cnt} new. Total: {len(all_rows)}")
        
        if last_ts >= end_ms:
            break
            
        if len(rows) < page_limit:
            break
            
        current_start = last_ts + 1
        time.sleep(0.1) 
        
    df = pd.DataFrame(all_rows)
    if not df.empty:
        df = df.sort_values('ts').reset_index(drop=True)
        df['open_time'] = df['ts'] 
        df['dt'] = pd.to_datetime(df['ts'], unit='ms')
    return df

def get_crypto_data(
    symbol: str,
    start_date: str,
    end_date: str,
    timeframe: str,
    project_root: Path
) -> pd.DataFrame:
    """
    Get crypto data from cache or fetch from Bybit.
    Cache location: data/raw/klines_cache/
    Filename format: {symbol}_{timeframe}_{start_date}_{end_date}.csv
    """
    cache_dir = project_root / "data" / "raw" / "klines_cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    
    filename = f"{symbol}_{timeframe}_{start_date}_{end_date}.csv"
    cache_path = cache_dir / filename
    
    # 1. Check Cache
    if cache_path.exists():
        logging.info(f"Loading data from cache: {cache_path}")
        df = pd.read_csv(cache_path)
        if 'open_time' not in df.columns and 'ts' in df.columns:
            df['open_time'] = df['ts']
        if 'ts' in df.columns:
            df['ts'] = df['ts'].astype(int)
            df['dt'] = pd.to_datetime(df['ts'], unit='ms')
        if 'open_time' in df.columns:
            df['open_time'] = df['open_time'].astype(int)
        if not df.empty and 'ts' in df.columns:
            df = df.sort_values('ts').reset_index(drop=True)
        return df
        
    # 2. Fetch if not cached
    logging.info(f"Cache miss. Fetching from Bybit: {symbol} {timeframe}")
    start_ts = int(pd.Timestamp(start_date, tz="UTC").timestamp() * 1000)
    end_ts = int(pd.Timestamp(end_date, tz="UTC").timestamp() * 1000)
    
    df = fetch_bybit_klines_range(
        symbol=symbol,
        interval=timeframe,
        start_ms=start_ts,
        end_ms=end_ts
    )
    
    # 3. Save to Cache
    if not df.empty:
        logging.info(f"Saving data to cache: {cache_path}")
        df.to_csv(cache_path, index=False)
        
    return df
