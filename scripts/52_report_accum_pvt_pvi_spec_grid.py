from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _run_one(
    *,
    script51: Path,
    symbols: str,
    interval: str,
    year_start: str,
    year_end: str,
    flow_indicator: str,
    extreme_seq_len: int,
    confluence_mode: str,
    cci_fast: int,
    cci_medium: int,
    cci_slow: int,
    cci_level: float,
    require_dmi_category: str,
    require_dmi_filter: str,
    enable_rebottom: bool,
    max_bottom_breaks: int,
    out_dir: Path,
) -> Path:
    cmd = [
        sys.executable,
        str(script51),
        "--symbols",
        str(symbols),
        "--interval",
        str(interval),
        "--year-start",
        str(year_start),
        "--year-end",
        str(year_end),
        "--flow-indicator",
        str(flow_indicator),
        "--extreme-seq-len",
        str(int(extreme_seq_len)),
        "--confluence-mode",
        str(confluence_mode),
        "--cci-fast",
        str(int(cci_fast)),
        "--cci-medium",
        str(int(cci_medium)),
        "--cci-slow",
        str(int(cci_slow)),
        "--cci-level",
        str(float(cci_level)),
        "--require-dmi-category",
        str(require_dmi_category),
        "--require-dmi-filter",
        str(require_dmi_filter),
        "--max-bottom-breaks",
        str(int(max_bottom_breaks)),
        "--out-dir",
        str(out_dir),
    ]
    if bool(enable_rebottom):
        cmd.append("--enable-rebottom")

    subprocess.run(cmd, cwd=str(PROJECT_ROOT), check=True)

    run_name = (
        f"accum_spec_{interval}_flow_{flow_indicator}_seq{int(extreme_seq_len)}"
        f"_cci{confluence_mode}_L{int(cci_level)}"
        f"_dmi_{require_dmi_category}_{require_dmi_filter}"
        f"_breaks{int(max_bottom_breaks)}"
    )
    return out_dir / run_name


def _safe_read_csv(path: Path) -> pd.DataFrame:
    try:
        return pd.read_csv(path)
    except Exception:
        return pd.DataFrame()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--symbols", default="LINKUSDT")
    ap.add_argument("--interval", default="1d")
    ap.add_argument("--year-start", default="2020-01-01")
    ap.add_argument("--year-end", default="2025-12-31")

    ap.add_argument("--flow-indicator", choices=["pvt", "pvt_pvi", "asi", "asi_pvt"], default="pvt")
    ap.add_argument("--extreme-seq-len", type=int, default=3)

    ap.add_argument("--confluence-mode", choices=["0", "2", "3"], default="2")
    ap.add_argument("--cci-fast", type=int, default=14)
    ap.add_argument("--cci-medium", type=int, default=30)
    ap.add_argument("--cci-slow", type=int, default=0)
    ap.add_argument("--cci-level", type=float, default=100.0)

    ap.add_argument("--require-dmi-category", default="any")
    ap.add_argument("--require-dmi-filter", default="any")

    ap.add_argument("--enable-rebottom", action="store_true")
    ap.add_argument("--breaks", default="0,1,2,3")

    ap.add_argument(
        "--out-dir",
        default=str(PROJECT_ROOT / "data" / "processed" / "accum_spec_grid_reports"),
    )

    args = ap.parse_args()

    breaks = [int(x.strip()) for x in str(args.breaks).split(",") if x.strip()]
    if not breaks:
        breaks = [0, 1, 2, 3]

    out_dir = Path(str(args.out_dir)).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    script51 = PROJECT_ROOT / "scripts" / "51_report_accum_pvt_pvi_spec.py"

    rows: list[dict[str, object]] = []

    for b in breaks:
        run_dir = _run_one(
            script51=script51,
            symbols=str(args.symbols),
            interval=str(args.interval),
            year_start=str(args.year_start),
            year_end=str(args.year_end),
            flow_indicator=str(args.flow_indicator),
            extreme_seq_len=int(args.extreme_seq_len),
            confluence_mode=str(args.confluence_mode),
            cci_fast=int(args.cci_fast),
            cci_medium=int(args.cci_medium),
            cci_slow=int(args.cci_slow),
            cci_level=float(args.cci_level),
            require_dmi_category=str(args.require_dmi_category),
            require_dmi_filter=str(args.require_dmi_filter),
            enable_rebottom=bool(args.enable_rebottom),
            max_bottom_breaks=int(b),
            out_dir=out_dir,
        )

        ev = _safe_read_csv(run_dir / "events.csv")
        sig = _safe_read_csv(run_dir / "signals.csv")
        st = _safe_read_csv(run_dir / "structure.csv")

        n_events = int(len(ev))
        n_signals = int(len(sig))
        n_acc_long = int((sig.get("kind") == "ACCUM_LONG").sum()) if len(sig) and "kind" in sig.columns else 0
        n_acc_short = int((sig.get("kind") == "ACCUM_SHORT").sum()) if len(sig) and "kind" in sig.columns else 0

        n_break = int((st.get("kind") == "BOTTOM_BREAK").sum()) if len(st) and "kind" in st.columns else 0
        n_rebottom = int((st.get("kind") == "REBOTTOM").sum()) if len(st) and "kind" in st.columns else 0

        rows.append(
            {
                "breaks": int(b),
                "events": n_events,
                "signals": n_signals,
                "accum_long": n_acc_long,
                "accum_short": n_acc_short,
                "bottom_break": n_break,
                "rebottom": n_rebottom,
                "run_dir": str(run_dir),
            }
        )

    df = pd.DataFrame(rows).sort_values(["breaks"]).reset_index(drop=True)
    out_csv = out_dir / "grid_summary.csv"
    df.to_csv(out_csv, index=False)

    print("\n=== GRID SUMMARY ===")
    print(df.to_string(index=False))
    print(f"\nWrote: {out_csv}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
