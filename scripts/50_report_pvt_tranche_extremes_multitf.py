from __future__ import annotations

import argparse
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]


@dataclass(frozen=True)
class Scenario:
    name: str
    interval: str
    cci_fast: int
    cci_medium: int
    lookahead: int


SCENARIOS = [
    Scenario("1d_v1_cci14_30", "1d", 14, 30, 365),
    Scenario("1d_v2_cci30_90", "1d", 30, 90, 365),
    Scenario("12h_v1_cci28_60", "12h", 28, 60, 730),
    Scenario("12h_v2_cci14_28", "12h", 14, 28, 730),
    Scenario("6h_cci28_56", "6h", 28, 56, 1460),
    Scenario("1h_cci72_168", "1h", 72, 168, 8760),
    Scenario("15m_cci24_288", "15m", 24, 288, 35040),
]


def _run_37(
    *,
    symbols: str,
    year_start: str,
    year_end: str,
    tranche_source: str,
    flow_indicator: str,
    max_bottom_breaks: int,
    extreme_seq_len: int,
    scenario: Scenario,
    export_dir: Path,
) -> None:
    script37 = PROJECT_ROOT / "scripts" / "37_verify_klinger_divergence_link_2024_2025.py"
    cmd = [
        sys.executable,
        str(script37),
        "--symbols",
        str(symbols),
        "--interval",
        str(scenario.interval),
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
        "--extreme-seq-len",
        str(int(extreme_seq_len)),
        "--lookahead",
        str(int(scenario.lookahead)),
        "--confluence-mode",
        "2",
        "--cci-fast",
        str(int(scenario.cci_fast)),
        "--cci-medium",
        str(int(scenario.cci_medium)),
        "--require-dmi-category",
        "any",
        "--print-campaigns",
        "none",
        "--export-dir",
        str(export_dir),
    ]
    subprocess.run(cmd, cwd=str(PROJECT_ROOT), check=True)


def _collect_campaigns(export_dir: Path) -> pd.DataFrame:
    paths = list(export_dir.glob("*_campaigns.csv"))
    dfs = []
    for p in paths:
        try:
            dfs.append(pd.read_csv(p))
        except Exception:
            continue
    if not dfs:
        return pd.DataFrame()
    return pd.concat(dfs, ignore_index=True)


def _summary(dfc: pd.DataFrame) -> dict[str, object]:
    if not len(dfc):
        return {
            "campaigns": 0,
            "confirmed": 0,
            "confirm_rate": float("nan"),
            "tranche_max_sum_p": float("nan"),
            "tranche_max_mean_p": float("nan"),
            "tranche_max_median_p": float("nan"),
            "tranche_max_p75_p": float("nan"),
            "tranche_max_p90_p": float("nan"),
            "campaign_mae_mean_p": float("nan"),
            "campaign_mae_median_p": float("nan"),
            "campaign_mdd_mean_p": float("nan"),
            "campaign_mdd_median_p": float("nan"),
        }

    conf = dfc[dfc["campaign_confirmed"] == True].copy()
    campaigns = int(len(dfc))
    confirmed = int(len(conf))
    confirm_rate = float(confirmed / campaigns) if campaigns else float("nan")

    tmax = pd.to_numeric(dfc.get("tranche_max_level_reached_p"), errors="coerce").astype(float).dropna()

    mae = pd.to_numeric(dfc.get("campaign_mae_p"), errors="coerce").astype(float).dropna()
    mdd = pd.to_numeric(dfc.get("campaign_mdd_p"), errors="coerce").astype(float).dropna()

    out = {
        "campaigns": campaigns,
        "confirmed": confirmed,
        "confirm_rate": confirm_rate,
        "tranche_max_sum_p": float(tmax.sum()) if len(tmax) else float("nan"),
        "tranche_max_mean_p": float(tmax.mean()) if len(tmax) else float("nan"),
        "tranche_max_median_p": float(tmax.median()) if len(tmax) else float("nan"),
        "tranche_max_p75_p": float(tmax.quantile(0.75)) if len(tmax) else float("nan"),
        "tranche_max_p90_p": float(tmax.quantile(0.90)) if len(tmax) else float("nan"),
        "campaign_mae_mean_p": float(mae.mean()) if len(mae) else float("nan"),
        "campaign_mae_median_p": float(mae.median()) if len(mae) else float("nan"),
        "campaign_mdd_mean_p": float(mdd.mean()) if len(mdd) else float("nan"),
        "campaign_mdd_median_p": float(mdd.median()) if len(mdd) else float("nan"),
    }
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--symbols", default="ETHUSDT,SOLUSDT,ADAUSDT,AVAXUSDT,BTCUSDT,LINKUSDT")
    ap.add_argument("--year-start", default="2020-01-01")
    ap.add_argument("--year-end", default="2025-12-31")
    ap.add_argument("--flow", choices=["klinger", "pvt", "pvt_pvi"], default="pvt")
    ap.add_argument("--flows", default="")
    ap.add_argument("--extreme-seq-lens", default="2,3,4")
    ap.add_argument("--max-bottom-breaks", type=int, default=0)
    ap.add_argument("--out-dir", default=str(PROJECT_ROOT / "data" / "processed" / "pvt_tranche_reports"))
    args = ap.parse_args()

    out_dir = Path(str(args.out_dir)).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    flows_s = str(args.flows).strip()
    flows = [str(args.flow)] if not flows_s else [x.strip() for x in flows_s.split(",") if x.strip()]
    flows = [f for f in flows if f in {"klinger", "pvt", "pvt_pvi"}]
    if not flows:
        flows = [str(args.flow)]

    seq_lens = [int(x.strip()) for x in str(args.extreme_seq_lens).split(",") if x.strip()]
    seq_lens = [x for x in seq_lens if x in {2, 3, 4}]
    if not seq_lens:
        seq_lens = [2, 3, 4]

    all_rows: list[dict[str, object]] = []

    for flow in flows:
        for seq_len in seq_lens:
            rows = []
            for sc in SCENARIOS:
                export_dir = (
                    out_dir
                    / f"runs_{args.year_start}_{args.year_end}"
                    / sc.name
                    / f"flow_{flow}"
                    / f"breaks_{int(args.max_bottom_breaks)}"
                    / f"seq_{int(seq_len)}"
                )
                export_dir.mkdir(parents=True, exist_ok=True)

                _run_37(
                    symbols=str(args.symbols),
                    year_start=str(args.year_start),
                    year_end=str(args.year_end),
                    tranche_source="macd_hist",
                    flow_indicator=str(flow),
                    max_bottom_breaks=int(args.max_bottom_breaks),
                    extreme_seq_len=int(seq_len),
                    scenario=sc,
                    export_dir=export_dir,
                )

                dfc = _collect_campaigns(export_dir)
                s = _summary(dfc)
                rows.append(
                    {
                        "flow": str(flow),
                        "extreme_seq_len": int(seq_len),
                        "scenario": sc.name,
                        "interval": sc.interval,
                        "cci_fast": sc.cci_fast,
                        "cci_medium": sc.cci_medium,
                        "lookahead": sc.lookahead,
                        **s,
                    }
                )

            res = pd.DataFrame(rows)
            all_rows.extend(rows)
            out_csv = out_dir / f"summary_{flow}_breaks{int(args.max_bottom_breaks)}_seq{int(seq_len)}.csv"
            res.to_csv(out_csv, index=False)

    full = pd.DataFrame(all_rows)
    print("\n=== Multi-flow tranche-extreme report ===")
    if len(full):
        print(full.to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
