from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import argparse
import datetime as dt

from libs.market_data.binance.dump_um_klines import dump_um_klines


def _parse_date(s: str) -> dt.date:
    return dt.date.fromisoformat(s)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ticker", default="LINKUSDT")
    parser.add_argument("--interval", default="4h")
    parser.add_argument("--start", default="2020-01-01", type=_parse_date)
    parser.add_argument("--end", default="2024-12-31", type=_parse_date, help="Inclusive end date")
    parser.add_argument("--root-dir", default="data/raw/binance_data_vision")
    parser.add_argument("--update-existing", action="store_true", help="Ne télécharge que les fichiers manquants")
    parser.add_argument("--skip-if-exists", action="store_true", help="Skip si le dataset complet existe déjà")
    args = parser.parse_args()

    # Vérification si le dataset complet existe déjà
    if args.skip_if_exists:
        from libs.market_data.binance.count_local_kline_csvs import count_local_kline_csvs
        
        existing_count = count_local_kline_csvs(
            root_dir=args.root_dir, 
            ticker=args.ticker, 
            interval=args.interval
        )
        
        # Pour les données 4h, Binance fournit des fichiers mensuels
        # On vérifie qu'on a assez de fichiers mensuels pour couvrir la période
        expected_months = ((args.end.year - args.start.year) * 12 + 
                          args.end.month - args.start.month + 1)
        
        # Seuil minimum : 80% des fichiers mensuels attendus
        min_required = int(expected_months * 0.8)
        
        if existing_count >= min_required:
            print(f"✅ Dataset quasi-complet déjà présent: {existing_count} fichiers mensuels")
            print(f"📂 Répertoire: {args.root_dir}")
            print(f"🪙 Ticker: {args.ticker}")
            print(f"⏱️ Interval: {args.interval}")
            print(f"📅 Période: {args.start} → {args.end} ({expected_months} mois attendus)")
            print("⏭️  Étape 1 ignorée (utilisation données existantes)")
            return 0
        else:
            print(f"⚠️  Dataset incomplet: {existing_count}/{min_required} fichiers mensuels minimum")
            print(f"📅 Période attendue: {args.start} → {args.end} ({expected_months} mois)")
            print("📥 Téléchargement des données manquantes...")

    dump_um_klines(
        root_dir=args.root_dir,
        ticker=args.ticker,
        interval=args.interval,
        date_start=args.start,
        date_end=args.end,
        update_existing=bool(args.update_existing),
    )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
