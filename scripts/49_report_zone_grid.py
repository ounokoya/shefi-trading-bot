from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _run_one(
    *,
    script37: Path,
    symbols: str,
    interval: str,
    year_start: str,
    year_end: str,
    tranche_source: str,
    flow_indicator: str,
    max_bottom_breaks: int,
    lookahead: int,
    export_dir: Path,
) -> None:
    cmd = [
        sys.executable,
        str(script37),
        "--symbols",
        str(symbols),
        "--interval",
        str(interval),
        "--year-start",
        str(year_start),
        "--year-end",
        str(year_end),
        "--tranche-source",
        str(tranche_source),
        "--flow-indicator",
        str(flow_indicator),
        "--max-bottom-breaks",
        str(int(max_bottom_breaks)),
        "--lookahead",
        str(int(lookahead)),
        "--print-campaigns",
        "none",
        "--export-dir",
        str(export_dir),
    ]
    subprocess.run(cmd, cwd=str(PROJECT_ROOT), check=True)


def _collect_campaigns(export_dir: Path, *, flow_indicator: str, max_bottom_breaks: int) -> pd.DataFrame:
    pat = f"*flow_{flow_indicator}_*breaks{int(max_bottom_breaks)}_campaigns.csv"
    paths = list(export_dir.glob(pat))
    if not paths:
        return pd.DataFrame()

    dfs = []
    for p in paths:
        try:
            dfs.append(pd.read_csv(p))
        except Exception:
            continue
    if not dfs:
        return pd.DataFrame()
    return pd.concat(dfs, ignore_index=True)


def _pct(x: float) -> str:
    return f"{100.0 * float(x):.1f}%"


def _summary(df: pd.DataFrame) -> dict[str, object]:
    if not len(df):
        return {
            "campaigns": 0,
            "confirmed": 0,
            "confirm_rate": float("nan"),
            "confirmed_ge_20p": 0,
            "confirmed_ge_30p": 0,
            "max_level_median": float("nan"),
            "confirm_level_median": float("nan"),
            "days_to_confirm_median": float("nan"),
        }

    conf = df[df["campaign_confirmed"] == True].copy()

    campaigns = int(len(df))
    confirmed = int(len(conf))
    confirm_rate = float(confirmed / campaigns) if campaigns else float("nan")

    maxlvl = pd.to_numeric(df.get("campaign_max_level_reached_p"), errors="coerce").astype(float)
    conflvl = pd.to_numeric(conf.get("campaign_confirm_level_p"), errors="coerce").astype(float)
    days = pd.to_numeric(conf.get("campaign_days_to_confirm"), errors="coerce").astype(float)

    confirmed_ge_20p = int((conflvl >= 20).sum()) if len(conflvl.dropna()) else 0
    confirmed_ge_30p = int((conflvl >= 30).sum()) if len(conflvl.dropna()) else 0

    return {
        "campaigns": campaigns,
        "confirmed": confirmed,
        "confirm_rate": confirm_rate,
        "confirmed_ge_20p": confirmed_ge_20p,
        "confirmed_ge_30p": confirmed_ge_30p,
        "max_level_median": float(maxlvl.median()) if len(maxlvl.dropna()) else float("nan"),
        "confirm_level_median": float(conflvl.median()) if len(conflvl.dropna()) else float("nan"),
        "days_to_confirm_median": float(days.median()) if len(days.dropna()) else float("nan"),
    }


def _level_histogram(values: pd.Series) -> str:
    s = pd.to_numeric(values, errors="coerce").dropna().astype(int)
    if not len(s):
        return ""
    vc = s.value_counts().sort_index()
    parts = []
    for k, v in vc.items():
        parts.append(f"{int(k)}:{int(v)}")
    return " ".join(parts)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--symbols", default="ETHUSDT,SOLUSDT,ADAUSDT,AVAXUSDT,BTCUSDT,LINKUSDT")
    ap.add_argument("--interval", default="1d")
    ap.add_argument("--year-start", default="2020-01-01")
    ap.add_argument("--year-end", default="2025-12-31")
    ap.add_argument("--tranche-source", default="macd_hist")
    ap.add_argument("--flows", default="klinger,pvt,nvi,pvi,pvt_pvi")
    ap.add_argument("--breaks", default="0,1,2,3")
    ap.add_argument("--lookahead", type=int, default=365)
    ap.add_argument("--out-dir", default=str(PROJECT_ROOT / "data" / "processed" / "zone_reports"))
    args = ap.parse_args()

    out_dir = Path(str(args.out_dir)).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    flows = [x.strip() for x in str(args.flows).split(",") if x.strip()]
    breaks = [int(x.strip()) for x in str(args.breaks).split(",") if x.strip()]

    script37 = PROJECT_ROOT / "scripts" / "37_verify_klinger_divergence_link_2024_2025.py"

    rows = []

    for flow in flows:
        for b in breaks:
            export_dir = out_dir / f"runs_{args.year_start}_{args.year_end}" / f"tr_{args.tranche_source}" / f"flow_{flow}" / f"breaks_{int(b)}"
            export_dir.mkdir(parents=True, exist_ok=True)

            _run_one(
                script37=script37,
                symbols=str(args.symbols),
                interval=str(args.interval),
                year_start=str(args.year_start),
                year_end=str(args.year_end),
                tranche_source=str(args.tranche_source),
                flow_indicator=str(flow),
                max_bottom_breaks=int(b),
                lookahead=int(args.lookahead),
                export_dir=export_dir,
            )

            dfc = _collect_campaigns(export_dir, flow_indicator=str(flow), max_bottom_breaks=int(b))
            summ = _summary(dfc)

            hist_confirm = ""
            hist_max = ""
            if len(dfc):
                hist_confirm = _level_histogram(dfc[dfc["campaign_confirmed"] == True]["campaign_confirm_level_p"])
                hist_max = _level_histogram(dfc["campaign_max_level_reached_p"])

            rows.append(
                {
                    "flow": str(flow),
                    "breaks": int(b),
                    **summ,
                    "confirm_level_hist": hist_confirm,
                    "max_level_hist": hist_max,
                }
            )

            out_csv = export_dir / "_summary.csv"
            pd.DataFrame([rows[-1]]).to_csv(out_csv, index=False)

    res = pd.DataFrame(rows)
    print("\n=== GRID SUMMARY ===")
    cols = [
        "flow",
        "breaks",
        "campaigns",
        "confirmed",
        "confirm_rate",
        "confirmed_ge_20p",
        "confirmed_ge_30p",
        "confirm_level_median",
        "max_level_median",
        "days_to_confirm_median",
    ]
    print(res[cols].to_string(index=False))

    print("\n=== HISTOGRAMS (confirm_level / max_level) ===")
    for r in rows:
        print(
            f"flow={r['flow']} breaks={r['breaks']} | confirm_level_hist: {r['confirm_level_hist']} | max_level_hist: {r['max_level_hist']}"
        )

    res.to_csv(out_dir / "grid_summary.csv", index=False)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
