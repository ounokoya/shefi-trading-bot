import sys
from pathlib import Path
import yaml
import argparse
import pandas as pd
import matplotlib.pyplot as plt
import logging
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
from libs.strategies.perp_hedge.brains.dmi_mfi_tf_15m import DMIMFITf15mBrain
from libs.strategies.perp_hedge.brains.vwma_dx_mfi_cci_mtf import VWMADxMfiCciMtfBrain
from libs.strategies.perp_hedge.mm import MoneyManagerConfig
from libs.strategies.perp_hedge.engine import BacktestEngine, BacktestConfig


def load_config(path: str) -> Dict[str, Any]:
    with open(path, 'r') as f:
        return yaml.safe_load(f)

def main():
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    
    parser = argparse.ArgumentParser()
    parser.add_argument('config_path', type=str, help='Path to YAML config file')
    args = parser.parse_args()
    
    cfg_data = load_config(args.config_path)
    
    bt_cfg_data = cfg_data.get('backtest', {})
    mm_cfg_data = cfg_data.get('money_manager', {})
    
    # Setup Config Objects
    mm_config = MoneyManagerConfig(**mm_cfg_data)
    
    bt_config = BacktestConfig(
        symbol=bt_cfg_data['symbol'],
        start_date=bt_cfg_data['start_date'],
        end_date=bt_cfg_data['end_date'],
        timeframe=bt_cfg_data['timeframe'],
        leverage=bt_cfg_data.get('leverage', 10.0),
        fee_rate=bt_cfg_data.get('fee_rate', 0.0015),
        liquidation_threshold_pct=bt_cfg_data.get('liquidation_threshold_pct', 0.05),
        mm_config=mm_config
    )
    
    # Fetch Data
    df = get_crypto_data(
        bt_config.symbol,
        bt_config.start_date,
        bt_config.end_date,
        bt_config.timeframe,
        PROJECT_ROOT
    )
    
    if df.empty:
        logging.error("No data fetched. Exiting.")
        return

    logging.info(f"Loaded {len(df)} rows.")

    try:
        if 'open_time' in df.columns and len(df) >= 3:
            ts_sorted = pd.to_numeric(df['open_time'], errors='coerce').dropna().astype(int).sort_values()
            if len(ts_sorted) >= 3:
                d_ms = ts_sorted.diff().dropna()
                inferred_ms = float(d_ms.median()) if not d_ms.empty else 0.0
                expected_tf = (bt_config.timeframe or '').strip().lower()
                expected_ms = 0.0
                if expected_tf.endswith('min') and expected_tf[:-3].isdigit():
                    expected_ms = float(int(expected_tf[:-3]) * 60_000)
                elif expected_tf.endswith('m') and expected_tf[:-1].isdigit():
                    expected_ms = float(int(expected_tf[:-1]) * 60_000)
                elif expected_tf.endswith('h') and expected_tf[:-1].isdigit():
                    expected_ms = float(int(expected_tf[:-1]) * 3_600_000)
                elif expected_tf.endswith('d') and expected_tf[:-1].isdigit():
                    expected_ms = float(int(expected_tf[:-1]) * 86_400_000)

                if expected_ms > 0 and inferred_ms > 0:
                    inferred_min = inferred_ms / 60_000.0
                    expected_min = expected_ms / 60_000.0
                    if abs(inferred_ms - expected_ms) > 1.0:
                        logging.warning(
                            f"Data timeframe mismatch? config={bt_config.timeframe!r} (~{expected_min:.2f}m) vs inferred~{inferred_min:.2f}m"
                        )
                    else:
                        logging.info(f"Data timeframe OK: {bt_config.timeframe!r} (~{expected_min:.2f}m)")
    except Exception:
        pass

    # Instantiate Brain and Engine
    brain_cfg = cfg_data.get('brain', {'type': 'dummy'})
    brain_type = brain_cfg.get('type', 'dummy')
    
    if brain_type == 'cci_1h_tf_5m' or brain_type == 'cci':
        logging.info("Using CCI 1H Brain (Tf 5m, Length 12)")
        brain = CCI1hTf5mBrain(
            source_df=df,
            source_tf=bt_config.timeframe
        )
    elif brain_type == 'cci_4h_tf_5m':
        logging.info("Using CCI 4H Brain (Tf 5m, Length 48)")
        brain = CCI4hTf5mBrain(source_df=df, source_tf=bt_config.timeframe)

    elif brain_type == 'cci_8h_tf_5m':
        logging.info("Using CCI 8H Brain (Tf 5m, Length 96)")
        brain = CCI8hTf5mBrain(source_df=df, source_tf=bt_config.timeframe)
        
    elif brain_type == 'cci_1h_4h_tf_5m':
        logging.info("Using CCI 1H+4H Brain (Tf 5m, Lengths 12 & 48)")
        brain = CCI1h4hTf5mBrain(source_df=df, source_tf=bt_config.timeframe)
        
    elif brain_type == 'cci_4h_8h_tf_5m':
        logging.info("Using CCI 4H+8H Brain (Tf 5m, Lengths 48 & 96)")
        brain = CCI4h8hTf5mBrain(source_df=df, source_tf=bt_config.timeframe)
        
    elif brain_type == 'cci_4h_tf_15m':
        logging.info("Using CCI 4H Brain (Tf 15m, Length 16)")
        brain = CCI4hTf15mBrain(source_df=df, source_tf=bt_config.timeframe)
        
    elif brain_type == 'cci_8h_tf_15m':
        logging.info("Using CCI 8H Brain (Tf 15m, Length 32)")
        brain = CCI8hTf15mBrain(source_df=df, source_tf=bt_config.timeframe)
        
    elif brain_type == 'cci_1d_tf_15m':
        logging.info("Using CCI 1D Brain (Tf 15m, Length 96)")
        brain = CCI1dTf15mBrain(source_df=df, source_tf=bt_config.timeframe)
        
    elif brain_type == 'cci_double_4h_8h_tf_15m':
        logging.info("Using CCI 4H+8H Brain (Tf 15m)")
        brain = CCIDouble4h8hTf15mBrain(source_df=df, source_tf=bt_config.timeframe)
        
    elif brain_type == 'cci_double_8h_1d_tf_15m':
        logging.info("Using CCI 8H+1D Brain (Tf 15m)")
        brain = CCIDouble8h1dTf15mBrain(source_df=df, source_tf=bt_config.timeframe)

    elif brain_type == 'dmi_mfi_tf_15m':
        logging.info("Using DMI+MFI Brain (Tf 15m)")
        if (bt_config.timeframe or '').lower().strip() not in {'15m', '15min'}:
            raise ValueError(f"brain.type='dmi_mfi_tf_15m' requires backtest.timeframe='15m' (got {bt_config.timeframe!r})")
        brain = DMIMFITf15mBrain(
            source_df=df,
            source_tf=bt_config.timeframe,
            dmi_period=int(brain_cfg.get('dmi_period', 14)),
            adx_smoothing=int(brain_cfg.get('adx_smoothing', 6)),
            maturity_mode=str(brain_cfg.get('maturity_mode', 'di_max')),
            adx_min_threshold=float(brain_cfg.get('adx_min_threshold', 20.0)),
            mfi_period=int(brain_cfg.get('mfi_period', 14)),
            strict_tf=bool(brain_cfg.get('strict_tf', True)),
        )

    elif brain_type == 'vwma_dx_mfi_cci_mtf':
        logging.info("Using VWMA+DX+MFI+CCI MTF Brain")
        brain = VWMADxMfiCciMtfBrain(
            source_df=df,
            source_tf=bt_config.timeframe,
            higher_tf=str(brain_cfg.get('higher_tf', '1h')),
            htf_close_opposite=bool(brain_cfg.get('htf_close_opposite', False)),
            vwma_fast_ltf=int(brain_cfg.get('vwma_fast_ltf', 12)),
            vwma_slow_ltf=int(brain_cfg.get('vwma_slow_ltf', 72)),
            vwma_fast_htf=int(brain_cfg.get('vwma_fast_htf', 12)),
            vwma_slow_htf=int(brain_cfg.get('vwma_slow_htf', 72)),
            dmi_period=int(brain_cfg.get('dmi_period', 14)),
            adx_smoothing=int(brain_cfg.get('adx_smoothing', 6)),
            mfi_period=int(brain_cfg.get('mfi_period', 14)),
            cci_period=int(brain_cfg.get('cci_period', 20)),
            mfi_high=float(brain_cfg.get('mfi_high', 60.0)),
            mfi_low=float(brain_cfg.get('mfi_low', 40.0)),
            cci_high=float(brain_cfg.get('cci_high', 80.0)),
            cci_low=float(brain_cfg.get('cci_low', -80.0)),
            ltf_gate_mode=str(brain_cfg.get('ltf_gate_mode', 'both')),
        )
        
    else:
        logging.info("Using Dummy Brain")
        brain = DummyBrain()

    engine = BacktestEngine(bt_config, brain, df)
    
    # Run
    engine.run()
    
    # Process Results
    equity_df = pd.DataFrame(engine.equity_history)
    trades_df = pd.DataFrame(engine.trades_history)
    
    # Output directory
    output_dir = PROJECT_ROOT / "data" / "processed" / "backtests" / "perp_hedge"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    output_prefix = output_dir / f"backtest_{bt_config.symbol}_{bt_config.timeframe}_{bt_config.start_date}_{bt_config.end_date}"
    
    # Save CSVs
    equity_csv = f"{output_prefix}_equity.csv"
    trades_csv = f"{output_prefix}_trades.csv"
    
    equity_df.to_csv(equity_csv, index=False)
    logging.info(f"Saved equity history to {equity_csv}")
    
    if not trades_df.empty:
        trades_df.to_csv(trades_csv, index=False)
        logging.info(f"Saved trades history to {trades_csv}")
        
        # Print summary stats
        total_trades = len(trades_df)
        win_trades = len(trades_df[trades_df['pnl_realized'] > 0]) if 'pnl_realized' in trades_df else 0
        logging.info(f"Total Trades: {total_trades}")
    
    # Plotting
    if not equity_df.empty:
        plt.figure(figsize=(12, 8))
        
        # Subplot 1: Equity & Wallet
        plt.subplot(2, 1, 1)
        plt.plot(pd.to_datetime(equity_df['timestamp'], unit='ms'), equity_df['equity'], label='Equity')
        plt.plot(pd.to_datetime(equity_df['timestamp'], unit='ms'), equity_df['wallet_balance'], label='Wallet Balance', linestyle='--')
        plt.title('Account Equity & Wallet Balance')
        plt.legend()
        plt.grid(True)
        
        # Subplot 2: Long vs Short Notional
        plt.subplot(2, 1, 2)
        plt.plot(pd.to_datetime(equity_df['timestamp'], unit='ms'), equity_df['long_notional'], label='Long Notional', color='green')
        plt.plot(pd.to_datetime(equity_df['timestamp'], unit='ms'), equity_df['short_notional'], label='Short Notional', color='red')
        plt.title('Position Notionals (Rebalancing)')
        plt.legend()
        plt.grid(True)
        
        plot_file = f"{output_prefix}_plot.png"
        plt.tight_layout()
        plt.savefig(plot_file)
        logging.info(f"Saved plot to {plot_file}")

    # --- Print Summary Report ---
    if not equity_df.empty:
        initial_capital = float(getattr(bt_config.mm_config, 'capital_usdt', 0.0) or 0.0)
        final_equity = float(equity_df.iloc[-1]['equity'])
        pnl_net = final_equity - initial_capital
        roi_pct = (pnl_net / initial_capital) * 100.0 if initial_capital > 0 else 0.0
        
        # Drawdown Calc
        equity_series = equity_df['equity']
        running_max = equity_series.cummax()
        drawdown = (equity_series - running_max) / running_max
        max_drawdown_pct = drawdown.min() * 100.0
        
        print("\n" + "="*40)
        print("          BACKTEST SUMMARY          ")
        print("="*40)
        print(f"Symbol           : {bt_config.symbol}")
        print(f"Period           : {bt_config.start_date} to {bt_config.end_date}")
        print(f"Initial Capital  : {initial_capital:.2f} USDT")
        print(f"Final Wallet     : {final_equity:.2f} USDT")
        print(f"Net PnL          : {pnl_net:+.2f} USDT ({roi_pct:+.2f}%)")
        print(f"Max Drawdown     : {max_drawdown_pct:.2f}%")
        
        if not trades_df.empty:
            total_trades = len(trades_df)
            
            realized_trades = pd.DataFrame()
            if 'pnl_realized' in trades_df.columns:
                 # Filter out 0 or NaN pnl_realized
                 realized_trades = trades_df[trades_df['pnl_realized'].fillna(0) != 0]

            if not realized_trades.empty:
                winners = realized_trades[realized_trades['pnl_realized'] > 0]
                win_rate = (len(winners) / len(realized_trades)) * 100.0
                print(f"Realized Trades  : {len(realized_trades)}")
                print(f"Win Rate         : {win_rate:.2f}%")
            print(f"Total MM Actions : {total_trades}")
        
        print("="*40 + "\n")

if __name__ == "__main__":
    main()
