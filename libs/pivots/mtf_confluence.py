from __future__ import annotations

import datetime as _dt
from typing import Any

from libs.pivots.pivot_registry import PivotRegistry


def format_dt_ms_utc(dt_ms: int | None) -> str:
    if dt_ms is None:
        return ""
    try:
        dt = _dt.datetime.fromtimestamp(int(dt_ms) / 1000.0, tz=_dt.timezone.utc)
        return dt.strftime("%Y-%m-%d %H:%M:%S UTC")
    except Exception:
        return ""


def zone_representative_event(
    reg: PivotRegistry,
    zone_id: str,
    *,
    prefer: str = "last",
) -> dict[str, Any] | None:
    z = reg.zones.get(str(zone_id))
    if not z:
        return None

    sel = z.get("selected_event_id")
    if sel is not None and str(sel) in reg.events:
        return dict(reg.events[str(sel)])

    ids = z.get("event_ids")
    if not isinstance(ids, list) or not ids:
        return None

    if str(prefer).lower() == "first":
        eid = str(ids[0])
    else:
        eid = str(ids[-1])

    ev = reg.events.get(eid)
    if not ev:
        return None
    return dict(ev)


def zone_ref_level(reg: PivotRegistry, zone_id: str) -> float | None:
    ev = zone_representative_event(reg, zone_id, prefer="first")
    if not ev:
        return None
    try:
        return float(ev["level"])
    except Exception:
        return None


def zone_touches(reg: PivotRegistry, zone_id: str) -> int:
    z = reg.zones.get(str(zone_id)) or {}
    ids = z.get("event_ids")
    if not isinstance(ids, list):
        return 0
    return int(len(ids))


def pct_dist(a: float, b: float) -> float:
    if b == 0:
        return float("inf")
    return float(abs((a / b) - 1.0))


def role_from_price(level: float, *, current_price: float) -> str | None:
    cp = float(current_price)
    lv = float(level)
    if not (cp == cp) or not (lv == lv) or cp == 0.0:
        return None
    if lv > cp:
        return "resistance"
    if lv < cp:
        return "support"
    return None


def match_zones(
    reg_a: PivotRegistry,
    reg_b: PivotRegistry,
    *,
    eps_mtf: float,
    current_price: float,
) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []

    for zid_a, za in reg_a.zones.items():
        ref_a = zone_ref_level(reg_a, str(zid_a))
        if ref_a is None:
            continue

        role_a = role_from_price(float(ref_a), current_price=float(current_price))
        if role_a not in {"support", "resistance"}:
            continue

        best: tuple[float, str, float] | None = None
        for zid_b, zb in reg_b.zones.items():
            ref_b = zone_ref_level(reg_b, str(zid_b))
            if ref_b is None:
                continue

            role_b = role_from_price(float(ref_b), current_price=float(current_price))
            if role_b != role_a:
                continue

            d = pct_dist(float(ref_a), float(ref_b))
            if d <= float(eps_mtf):
                if best is None or d < best[0]:
                    best = (float(d), str(zid_b), float(ref_b))

        if best is None:
            continue

        d, zid_b, ref_b = best
        out.append(
            {
                "role": str(role_a),
                "zone_a": str(zid_a),
                "zone_b": str(zid_b),
                "ref_a": float(ref_a),
                "ref_b": float(ref_b),
                "d_abs": float(d),
                "touches_a": zone_touches(reg_a, str(zid_a)),
                "touches_b": zone_touches(reg_b, str(zid_b)),
            }
        )

    out.sort(key=lambda r: (str(r["role"]), float(r["d_abs"])))
    return out


def build_triple_from_pairs(
    *,
    pairs_low_mid: list[dict[str, Any]],
    pairs_mid_high: list[dict[str, Any]],
    mid_key: str,
    eps_mtf: float,
) -> list[dict[str, Any]]:
    low_by_mid: dict[str, dict[str, Any]] = {}
    for p in pairs_low_mid:
        low_by_mid[str(p["zone_b"])] = p

    high_by_mid: dict[str, dict[str, Any]] = {}
    for p in pairs_mid_high:
        high_by_mid[str(p["zone_a"])] = p

    out: list[dict[str, Any]] = []
    for mid_zid, p_lm in low_by_mid.items():
        p_mh = high_by_mid.get(str(mid_zid))
        if p_mh is None:
            continue
        if str(p_lm.get("role")) != str(p_mh.get("role")):
            continue

        out.append(
            {
                "role": str(p_lm["role"]),
                "eps_mtf": float(eps_mtf),
                "members": {
                    "low": str(p_lm["zone_a"]),
                    mid_key: str(mid_zid),
                    "high": str(p_mh["zone_b"]),
                },
                "d_low_mid": float(p_lm["d_abs"]),
                "d_mid_high": float(p_mh["d_abs"]),
            }
        )

    out.sort(key=lambda r: (str(r["role"]), float(r["d_low_mid"]) + float(r["d_mid_high"])))
    return out
