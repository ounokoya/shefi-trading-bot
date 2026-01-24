from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


def _summarize_legs(legs: pd.DataFrame, *, target: float) -> dict[str, object]:
    legs = legs.copy()

    legs["ret_pct"] = pd.to_numeric(legs.get("ret_pct"), errors="coerce")
    legs["dd_float_max"] = pd.to_numeric(legs.get("dd_float_max"), errors="coerce")
    legs["dd_float_mean"] = pd.to_numeric(legs.get("dd_float_mean"), errors="coerce")

    meets = (legs["ret_pct"].fillna(float("nan")) >= float(target)).fillna(False)

    out: dict[str, object] = {
        "n_legs": int(len(legs)),
        "target": float(target),
        "capture_sum": float(legs["ret_pct"].dropna().sum()),
        "capture_mean": float(legs["ret_pct"].dropna().mean()) if legs["ret_pct"].notna().any() else None,
        "capture_sum_meeting_target": float(legs.loc[meets, "ret_pct"].dropna().sum()),
        "capture_mean_meeting_target": float(legs.loc[meets, "ret_pct"].dropna().mean())
        if legs.loc[meets, "ret_pct"].notna().any()
        else None,
        "winrate_meeting_target": float(meets.mean()) if len(meets) else None,
        "dd_max": float(legs["dd_float_max"].dropna().max()) if legs["dd_float_max"].notna().any() else None,
        "dd_mean": float(legs["dd_float_mean"].dropna().mean()) if legs["dd_float_mean"].notna().any() else None,
    }

    if "leg" in legs.columns and "leg_side" in legs.columns:
        g = (
            legs.assign(meets_target=meets.astype(int))
            .groupby(["leg", "leg_side"], dropna=False)
            .agg(
                n_legs=("ret_pct", "size"),
                capture_sum=("ret_pct", "sum"),
                capture_mean=("ret_pct", "mean"),
                winrate=("meets_target", "mean"),
                dd_max=("dd_float_max", "max"),
                dd_mean=("dd_float_mean", "mean"),
            )
            .reset_index()
        )
        out["by_leg"] = g.to_dict(orient="records")

    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--perfect-legs-csv", required=True)
    ap.add_argument("--candidate-legs-csv", required=True)
    ap.add_argument("--target", type=float, default=0.007)
    args = ap.parse_args()

    p = Path(str(args.perfect_legs_csv))
    c = Path(str(args.candidate_legs_csv))

    perfect = pd.read_csv(p)
    candidate = pd.read_csv(c)

    sp = _summarize_legs(perfect, target=float(args.target))
    sc = _summarize_legs(candidate, target=float(args.target))

    out = {
        "perfect": sp,
        "candidate": sc,
        "delta": {
            "capture_sum": (sc["capture_sum"] - sp["capture_sum"]) if (sc.get("capture_sum") is not None and sp.get("capture_sum") is not None) else None,
            "capture_sum_meeting_target": (sc["capture_sum_meeting_target"] - sp["capture_sum_meeting_target"])
            if (sc.get("capture_sum_meeting_target") is not None and sp.get("capture_sum_meeting_target") is not None)
            else None,
            "winrate_meeting_target": (sc["winrate_meeting_target"] - sp["winrate_meeting_target"])
            if (sc.get("winrate_meeting_target") is not None and sp.get("winrate_meeting_target") is not None)
            else None,
            "dd_max": (sc["dd_max"] - sp["dd_max"]) if (sc.get("dd_max") is not None and sp.get("dd_max") is not None) else None,
        },
    }

    print(json.dumps(out, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
