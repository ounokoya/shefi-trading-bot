import sys
from pathlib import Path
import pandas as pd
import logging

# Setup Project Root
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from libs.data_loader import get_crypto_data
from libs.strategies.perp_hedge.brains.cci_1h_tf_5m import CCI1hTf5mBrain
from libs.strategies.perp_hedge.mm import MoneyManagerConfig
from libs.strategies.perp_hedge.engine import BacktestEngine, BacktestConfig

def run_debug_liquidation():
    # Setup Logging to console
    logging.basicConfig(level=logging.INFO, format='%(message)s')
    
    symbol = "LINKUSDT"
    timeframe = "5m"
    start_date = "2024-01-01"
    end_date = "2024-12-31"
    
    print(f"Loading data for {symbol} ({start_date} to {end_date})...")
    df = get_crypto_data(symbol, start_date, end_date, timeframe, PROJECT_ROOT)
    
    if df.empty:
        print("No data found.")
        return

    # Config identique au script de comparaison
    mm_config = MoneyManagerConfig(
        capital_usdt=1000.0,
        initial_long_usdt=100.0,
        initial_short_usdt=100.0,
        max_initial_invest_usdt=600.0,
        gap_mode="leveq",
        max_notional_gap_pct=0.85,
        min_leg_notional_usdt=25.0,
        gap_threshold_pct=0.25,
        min_step_usdt=25.0,
        max_step_usdt=150.0,
        cooldown_bars=3,
        leverage=10.0
    )
    
    bt_config = BacktestConfig(
        symbol=symbol,
        start_date=start_date,
        end_date=end_date,
        timeframe=timeframe,
        leverage=10.0,
        fee_rate=0.0015,
        liquidation_threshold_pct=0.05,
        mm_config=mm_config
    )
    
    print("\nRunning Backtest for CCI 1H (Standard)...")
    # Utilisation du cerveau CCI 1H Standard (sans filtre) pour voir le pire cas ou un cas typique
    brain = CCI1hTf5mBrain(source_df=df, source_tf=timeframe, filter_mode="none")
    
    engine = BacktestEngine(bt_config, brain, df)
    engine.run()

    output_dir = PROJECT_ROOT / "data" / "processed" / "backtests" / "perp_hedge"
    output_dir.mkdir(parents=True, exist_ok=True)

    output_prefix = output_dir / f"debug_liquidation_{symbol}_{start_date}_{end_date}_{timeframe}"

    equity_df = pd.DataFrame(engine.equity_history)
    trades_df = pd.DataFrame(engine.trades_history)

    equity_csv = f"{output_prefix}_equity.csv"
    trades_csv = f"{output_prefix}_trades.csv"

    if not equity_df.empty:
        equity_df.to_csv(equity_csv, index=False)
        print(f"Saved equity history to {equity_csv}")

    trades_df.to_csv(trades_csv, index=False)
    print(f"Saved trades history to {trades_csv}")
    
    if engine.is_liquidated:
        print("\n>>> ANALYSE LIQUIDATION TERMINÉE <<<")
    else:
        print("\nPas de liquidation sur ce run.")

if __name__ == "__main__":
    run_debug_liquidation()
