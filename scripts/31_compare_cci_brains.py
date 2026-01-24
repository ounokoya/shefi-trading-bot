import sys
from pathlib import Path
import pandas as pd
import logging
import time
from typing import Dict, Any

# Setup Project Root
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from libs.data_loader import get_crypto_data
from libs.strategies.perp_hedge.models import PositionSide
from libs.strategies.perp_hedge.brain import DummyBrain
from libs.strategies.perp_hedge.brains.cci_1h_tf_5m import CCI1hTf5mBrain
from libs.strategies.perp_hedge.brains.cci_4h_tf_5m import CCI4hTf5mBrain
from libs.strategies.perp_hedge.brains.cci_8h_tf_5m import CCI8hTf5mBrain
from libs.strategies.perp_hedge.brains.cci_1h_4h_tf_5m import CCI1h4hTf5mBrain
from libs.strategies.perp_hedge.brains.cci_4h_8h_tf_5m import CCI4h8hTf5mBrain
from libs.strategies.perp_hedge.brains.cci_4h_tf_15m import CCI4hTf15mBrain
from libs.strategies.perp_hedge.brains.cci_8h_tf_15m import CCI8hTf15mBrain
from libs.strategies.perp_hedge.brains.cci_1d_tf_15m import CCI1dTf15mBrain
from libs.strategies.perp_hedge.brains.cci_double_4h_8h_tf_15m import CCIDouble4h8hTf15mBrain
from libs.strategies.perp_hedge.brains.cci_double_8h_1d_tf_15m import CCIDouble8h1dTf15mBrain
from libs.strategies.perp_hedge.mm import MoneyManagerConfig
from libs.strategies.perp_hedge.engine import BacktestEngine, BacktestConfig

def run_comparison():
    # Configuration
    symbol = "LINKUSDT"
    timeframe = "5m"
    start_date = "2024-01-01"
    end_date = "2024-12-31"
    
    # MM Config
    mm_config = MoneyManagerConfig(
        capital_usdt=1000.0,
        initial_long_usdt=100.0,
        initial_short_usdt=100.0,
        max_initial_invest_usdt=600.0,
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
    
    # Fetch Data with Cache
    df = get_crypto_data(symbol, start_date, end_date, timeframe, PROJECT_ROOT)
    
    if df.empty:
        print("No data!")
        return

    brains_to_test_config = [("Dummy (No Filter)", DummyBrain, {"source_df": df})]
    
    # List of Brain Classes and their display names
    brain_classes = [
        ("CCI 1H (12 on 5m)", CCI1hTf5mBrain),
        ("CCI 4H (48 on 5m)", CCI4hTf5mBrain),
        ("CCI 8H (96 on 5m)", CCI8hTf5mBrain),
        ("CCI 1H+4H (5m)", CCI1h4hTf5mBrain),
        ("CCI 4H+8H (5m)", CCI4h8hTf5mBrain),
        ("CCI 4H (16 on 15m)", CCI4hTf15mBrain),
        ("CCI 8H (32 on 15m)", CCI8hTf15mBrain),
        ("CCI 1D (96 on 15m)", CCI1dTf15mBrain),
        ("CCI 4H+8H (15m)", CCIDouble4h8hTf15mBrain),
        ("CCI 8H+1D (15m)", CCIDouble8h1dTf15mBrain)
    ]
    
    # Add Standard, KVO Filter, Signal Filter versions to config list
    for name, brain_cls in brain_classes:
        # Standard
        brains_to_test_config.append((f"{name} [Standard]", brain_cls, {"source_df": df, "source_tf": timeframe, "filter_mode": "none"}))
        # Klinger KVO
        brains_to_test_config.append((f"{name} [KVO Filter]", brain_cls, {"source_df": df, "source_tf": timeframe, "filter_mode": "kvo"}))
        # Klinger Signal
        brains_to_test_config.append((f"{name} [Signal Filter]", brain_cls, {"source_df": df, "source_tf": timeframe, "filter_mode": "signal"}))
    
    results = []
    
    print(f"\nRunning Comparison on {symbol} ({start_date} to {end_date})")
    print("-" * 60)
    
    for name, brain_cls, kwargs in brains_to_test_config:
        print(f"Running {name}...", end="", flush=True)
        try:
            # Instantiate Brain Here
            if name == "Dummy (No Filter)":
                brain = brain_cls() # DummyBrain takes no args in this script's context usually, or we fix it
            else:
                brain = brain_cls(**kwargs)
                
            engine = BacktestEngine(bt_config, brain, df)
            engine.run() 
            
            equity_df = pd.DataFrame(engine.equity_history)
            trades_df = pd.DataFrame(engine.trades_history)
            
            if equity_df.empty:
                print(" Error: No equity history.")
                continue

            initial = equity_df.iloc[0]['equity']
            final = equity_df.iloc[-1]['equity']
            pnl = final - initial
            pnl_pct = (pnl / initial) * 100
            
            # Max Drawdown
            eq = equity_df['equity']
            dd = (eq - eq.cummax()) / eq.cummax() * 100
            max_dd = dd.min()
            
            count = len(trades_df)
            win_rate = 0
            if not trades_df.empty and 'pnl_realized' in trades_df:
                realized = trades_df[trades_df['pnl_realized'].fillna(0) != 0]
                if not realized.empty:
                    wins = realized[realized['pnl_realized'] > 0]
                    win_rate = (len(wins) / len(realized)) * 100
            
            results.append({
                "Brain": name,
                "PnL $": pnl,
                "PnL %": pnl_pct,
                "Max DD %": max_dd,
                "Trades": count,
                "WinRate %": win_rate
            })
            print(" Done.")
        except Exception as e:
            print(f" Failed: {e}")
            import traceback
            traceback.print_exc()

    # Display Results

    # Display Results
    res_df = pd.DataFrame(results)
    print("\n" + "="*80)
    print("COMPARATIVE RESULTS")
    print("="*80)
    print(res_df.to_string(index=False, float_format=lambda x: "{:.2f}".format(x)))
    print("="*80)

if __name__ == "__main__":
    # Mute existing logging to clean output
    logging.getLogger().setLevel(logging.WARNING)
    run_comparison()
