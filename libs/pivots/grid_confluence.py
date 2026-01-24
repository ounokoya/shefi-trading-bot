from __future__ import annotations

import math
from typing import Any

from libs.pivots.mtf_confluence import role_from_price, zone_representative_event
from libs.pivots.pivot_registry import PivotRegistry


def _pct_dist(a: float, b: float) -> float:
    if b == 0:
        return float("inf")
    return float(abs((float(a) / float(b)) - 1.0))


def _tf_ms_from_registry(reg: PivotRegistry, tf_fallback: str) -> int:
    v = reg.meta.get("tf_ms")
    if isinstance(v, int) and v > 0:
        return int(v)

    s = str(tf_fallback).strip().lower()
    if s.endswith("m") and s[:-1].isdigit():
        return int(s[:-1]) * 60_000
    if s.endswith("h") and s[:-1].isdigit():
        return int(s[:-1]) * 3_600_000
    if s.endswith("d") and s[:-1].isdigit():
        return int(s[:-1]) * 86_400_000
    if s == "d":
        return 86_400_000
    raise ValueError(f"unsupported tf: {tf_fallback}")


def grid_key_for_level(*, level: float, anchor_price: float, grid_pct: float) -> int:
    g = float(grid_pct)
    if g <= 0:
        raise ValueError("grid_pct must be > 0")
    a = float(anchor_price)
    if a <= 0:
        raise ValueError("anchor_price must be > 0")

    x = float(level) / a
    if x <= 0:
        return 0

    step = math.log1p(g)
    return int(round(math.log(x) / step))


def grid_level_from_key(*, key: int, anchor_price: float, grid_pct: float) -> float:
    g = float(grid_pct)
    a = float(anchor_price)
    return float(a * ((1.0 + g) ** int(key)))


def _collect_zone_candidates(
    *,
    current_price: float,
    now_ts_ms: int,
    regs_by_tf: dict[str, PivotRegistry],
    tf_importance: dict[str, int],
) -> list[dict[str, Any]]:
    candidates: list[dict[str, Any]] = []
    for tf, reg in regs_by_tf.items():
        tf_ms = _tf_ms_from_registry(reg, str(tf))
        imp = int(tf_importance.get(str(tf), 1))

        for zid in reg.zones.keys():
            ev = zone_representative_event(reg, str(zid), prefer="last")
            if not ev:
                continue
            try:
                level = float(ev["level"])
            except Exception:
                continue
            if level <= 0:
                continue

            role_now = role_from_price(float(level), current_price=float(current_price))
            if role_now not in {"support", "resistance"}:
                continue

            dt_ms = ev.get("dt_ms")
            if not isinstance(dt_ms, int):
                continue
            bars_ago = int(max(0, (int(now_ts_ms) - int(dt_ms)) // int(tf_ms)))

            instant = ev.get("instant") or {}
            tags = instant.get("tags_ccis") or []
            tags_norm = [str(x).strip().lower() for x in tags if str(x).strip()]
            cci_weight = int(len(set(tags_norm)))

            candidates.append(
                {
                    "tf": str(tf),
                    "tf_importance": int(imp),
                    "role": str(role_now),
                    "zone_id": str(zid),
                    "event_id": str(ev.get("event_id") or ""),
                    "dt_ms": int(dt_ms),
                    "dt": str(ev.get("dt") or ""),
                    "bars_ago": int(bars_ago),
                    "level": float(level),
                    "cci_tags": tags_norm,
                    "cci_weight": int(cci_weight),
                }
            )

    return candidates


def _cluster_items(
    *,
    items: list[dict[str, Any]],
    radius_pct: float,
    padding_pct: float,
) -> list[dict[str, Any]]:
    if not items:
        return []

    r = float(radius_pct)
    p = float(padding_pct)
    if r <= 0:
        raise ValueError("radius_pct must be > 0")
    if p < 0:
        raise ValueError("padding_pct must be >= 0")

    n = len(items)
    parent = list(range(n))

    def _find(i: int) -> int:
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    def _union(a: int, b: int) -> None:
        ra = _find(a)
        rb = _find(b)
        if ra != rb:
            parent[rb] = ra

    levels = [float(x["level"]) for x in items]
    for i in range(n):
        for j in range(i + 1, n):
            if _pct_dist(levels[i], levels[j]) <= r:
                _union(i, j)

    groups: dict[int, list[dict[str, Any]]] = {}
    for i, it in enumerate(items):
        groups.setdefault(_find(i), []).append(it)

    def _zone_from_members(members: list[dict[str, Any]]) -> dict[str, Any]:
        ws = [float(max(1, int(x.get("tf_importance") or 1))) for x in members]
        lv = [float(x["level"]) for x in members]
        wsum = float(sum(ws))
        center = float(sum(l * w for l, w in zip(lv, ws)) / wsum) if wsum > 0 else float(sum(lv) / len(lv))
        return {
            "role": str(members[0]["role"]),
            "tf": str(members[0]["tf"]),
            "center_level": float(center),
            "bounds": {"lower": float(min(lv)), "upper": float(max(lv))},
            "weight_sum": float(wsum),
            "members": [dict(x) for x in members],
        }

    zones = [_zone_from_members(m) for m in groups.values()]

    if p <= 0 or len(zones) <= 1:
        zones.sort(key=lambda z: float(z["center_level"]))
        return zones

    merged = True
    while merged and len(zones) > 1:
        merged = False
        zones.sort(key=lambda z: float(z["center_level"]))
        out: list[dict[str, Any]] = []
        i = 0
        while i < len(zones):
            cur = zones[i]
            j = i + 1
            while j < len(zones):
                nxt = zones[j]
                if _pct_dist(float(cur["center_level"]), float(nxt["center_level"])) <= p:
                    merged = True
                    cur_members = list(cur.get("members") or []) + list(nxt.get("members") or [])
                    cur = _zone_from_members(cur_members)
                    j += 1
                else:
                    break
            out.append(cur)
            i = j
        zones = out

    zones.sort(key=lambda z: float(z["center_level"]))
    return zones


def _nearest_member(*, base_level: float, members: list[dict[str, Any]], max_pct: float) -> dict[str, Any] | None:
    if not members:
        return None
    eligible = [m for m in members if _pct_dist(float(m["level"]), float(base_level)) <= float(max_pct)]
    if not eligible:
        return None
    return dict(min(eligible, key=lambda m: (abs(float(m["level"]) - float(base_level)), int(m.get("bars_ago") or 0))))


def _build_hierarchical_zones(
    *,
    current_price: float,
    candidates: list[dict[str, Any]],
    levels: list[dict[str, Any]],
    keep_top2_5m: bool,
) -> list[dict[str, Any]]:
    if len(levels) != 3:
        raise ValueError("levels must have exactly 3 entries: macro/context/execution")

    macro = dict(levels[0])
    context = dict(levels[1])
    exec_ = dict(levels[2])

    macro_tf = str(macro["tf"])
    context_tf = str(context["tf"])
    exec_tf = str(exec_["tf"])

    macro_r = float(macro["radius_pct"])
    macro_p = float(macro["padding_pct"])
    context_r = float(context["radius_pct"])
    context_p = float(context["padding_pct"])
    exec_r = float(exec_["radius_pct"])
    exec_p = float(exec_["padding_pct"])

    expand_factor = 2.0
    max_expand_steps = 3

    def _items_in_band(*, tf: str, role: str, center_level: float, band_pct: float) -> list[dict[str, Any]]:
        lo = float(center_level) * (1.0 - float(band_pct))
        hi = float(center_level) * (1.0 + float(band_pct))
        return [
            c
            for c in candidates
            if str(c["tf"]) == str(tf)
            and str(c["role"]) == str(role)
            and float(c["level"]) >= lo
            and float(c["level"]) <= hi
        ]

    out_macro: list[dict[str, Any]] = []
    for role in ["support", "resistance"]:
        lo = float(current_price) * (1.0 - macro_r)
        hi = float(current_price) * (1.0 + macro_r)
        macro_items = [
            c
            for c in candidates
            if str(c["tf"]) == macro_tf
            and str(c["role"]) == role
            and float(c["level"]) >= lo
            and float(c["level"]) <= hi
        ]
        macro_zones = _cluster_items(items=macro_items, radius_pct=float(macro_r), padding_pct=float(macro_p))

        for mz in macro_zones:
            mz["subzones"] = []
            mz_center = float(mz.get("center_level") or 0.0)

            out_ctx: list[dict[str, Any]] = []
            used_macro_child_r: float | None = None
            for step_m in range(int(max_expand_steps) + 1):
                macro_child_r = float(macro_r) * (float(expand_factor) ** float(step_m))
                ctx_items = _items_in_band(tf=str(context_tf), role=str(role), center_level=float(mz_center), band_pct=float(macro_child_r))
                ctx_zones = _cluster_items(items=ctx_items, radius_pct=float(context_r), padding_pct=float(context_p))

                out_ctx = []
                for cz in ctx_zones:
                    cz["subzones"] = []
                    cz_center = float(cz.get("center_level") or 0.0)

                    out_ex: list[dict[str, Any]] = []
                    used_context_child_r: float | None = None
                    for step_c in range(int(max_expand_steps) + 1):
                        context_child_r = float(context_r) * (float(expand_factor) ** float(step_c))
                        ex_items = _items_in_band(
                            tf=str(exec_tf),
                            role=str(role),
                            center_level=float(cz_center),
                            band_pct=float(context_child_r),
                        )
                        ex_zones = _cluster_items(items=ex_items, radius_pct=float(exec_r), padding_pct=float(exec_p))

                        out_ex = []
                        for ez in ex_zones:
                            ez["subzones"] = []
                            exec_members = list(ez.get("members") or [])
                            exec_members.sort(key=lambda m: (int(m.get("bars_ago") or 0), -int(m.get("dt_ms") or 0)))
                            top2 = exec_members[:2] if keep_top2_5m else exec_members[:1]
                            picks: list[dict[str, Any]] = []
                            for p5 in top2:
                                p1 = _nearest_member(
                                    base_level=float(p5["level"]),
                                    members=list(cz.get("members") or []),
                                    max_pct=float(context_r),
                                )
                                p4 = _nearest_member(
                                    base_level=float(p5["level"]),
                                    members=list(mz.get("members") or []),
                                    max_pct=float(macro_r),
                                )
                                if p1 is None or p4 is None:
                                    continue
                                local_w = (
                                    int(p5.get("cci_weight") or 0)
                                    + int(p1.get("cci_weight") or 0)
                                    + int(p4.get("cci_weight") or 0)
                                )
                                global_w = (
                                    int(p5.get("tf_importance") or 0)
                                    + int(p1.get("tf_importance") or 0)
                                    + int(p4.get("tf_importance") or 0)
                                )
                                picks.append(
                                    {
                                        "members": {"5m": dict(p5), "1h": dict(p1), "4h": dict(p4)},
                                        "score": {"importance_global": int(global_w), "importance_local": int(local_w)},
                                    }
                                )
                            ez["picks"] = picks
                            if picks:
                                out_ex.append(ez)

                        if out_ex:
                            used_context_child_r = float(context_child_r)
                            break

                    if out_ex:
                        if used_context_child_r is not None and used_context_child_r != float(context_r):
                            cz["child_search_radius_pct"] = float(used_context_child_r)
                        cz["subzones"] = out_ex
                        cz_levels = [float(m["level"]) for m in (cz.get("members") or [])]
                        cz_levels.extend(float(m["level"]) for ex in out_ex for m in (ex.get("members") or []))
                        if cz_levels:
                            cz["bounds"] = {"lower": float(min(cz_levels)), "upper": float(max(cz_levels))}
                        out_ctx.append(cz)

                if out_ctx:
                    used_macro_child_r = float(macro_child_r)
                    break

            if out_ctx:
                if used_macro_child_r is not None and used_macro_child_r != float(macro_r):
                    mz["child_search_radius_pct"] = float(used_macro_child_r)
                mz["subzones"] = out_ctx
                mz_levels = [float(m["level"]) for m in (mz.get("members") or [])]
                mz_levels.extend(float(m["level"]) for cx in out_ctx for m in (cx.get("members") or []))
                mz_levels.extend(
                    float(m["level"]) for cx in out_ctx for ex in (cx.get("subzones") or []) for m in (ex.get("members") or [])
                )
                if mz_levels:
                    mz["bounds"] = {"lower": float(min(mz_levels)), "upper": float(max(mz_levels))}
                out_macro.append(mz)

    out_macro.sort(key=lambda z: (str(z.get("role") or ""), float(z.get("center_level") or 0.0)))
    return out_macro


def build_grid_confluence(
    *,
    symbol: str,
    current_price: float,
    now_ts_ms: int,
    grid_pct: float,
    regs_by_tf: dict[str, PivotRegistry],
    tf_importance: dict[str, int] | None = None,
    keep_top2_5m: bool = True,
    mode: str = "grid",
    zones_cfg: dict[str, dict[str, Any]] | None = None,
) -> dict[str, Any]:
    tfi = dict(tf_importance or {"5m": 1, "1h": 2, "4h": 3})

    def _cfg_float(d: dict[str, Any], key: str, default: float) -> float:
        v = d.get(key)
        return float(default) if v is None else float(v)

    mm = str(mode).strip().lower()
    if mm == "zones":
        cfg = dict(zones_cfg or {})
        macro_cfg = dict(cfg.get("macro") or {"tf": "4h", "radius_pct": float(grid_pct), "padding_pct": float(grid_pct)})
        context_cfg = dict(cfg.get("context") or {"tf": "1h", "radius_pct": float(grid_pct), "padding_pct": float(grid_pct)})
        exec_cfg = dict(cfg.get("execution") or {"tf": "5m", "radius_pct": float(grid_pct), "padding_pct": float(grid_pct)})

        default_radius = float(grid_pct) if float(grid_pct) > 0 else None

        macro_r = _cfg_float(macro_cfg, "radius_pct", float(default_radius or 0.0))
        context_r = _cfg_float(context_cfg, "radius_pct", float(default_radius or 0.0))
        exec_r = _cfg_float(exec_cfg, "radius_pct", float(default_radius or 0.0))
        if macro_r <= 0 or context_r <= 0 or exec_r <= 0:
            raise ValueError("mode='zones' requires radius_pct per level (macro/context/execution) or grid_pct>0 as fallback")

        macro_p = _cfg_float(macro_cfg, "padding_pct", float(macro_r))
        context_p = _cfg_float(context_cfg, "padding_pct", float(context_r))
        exec_p = _cfg_float(exec_cfg, "padding_pct", float(exec_r))

        candidates = _collect_zone_candidates(
            current_price=float(current_price),
            now_ts_ms=int(now_ts_ms),
            regs_by_tf=regs_by_tf,
            tf_importance=tfi,
        )
        zones = _build_hierarchical_zones(
            current_price=float(current_price),
            candidates=candidates,
            levels=[
                {"tf": str(macro_cfg.get("tf") or "4h"), "radius_pct": float(macro_r), "padding_pct": float(macro_p)},
                {"tf": str(context_cfg.get("tf") or "1h"), "radius_pct": float(context_r), "padding_pct": float(context_p)},
                {"tf": str(exec_cfg.get("tf") or "5m"), "radius_pct": float(exec_r), "padding_pct": float(exec_p)},
            ],
            keep_top2_5m=bool(keep_top2_5m),
        )

        return {
            "meta": {
                "symbol": str(symbol),
                "mode": "zones",
                "current_price": float(current_price),
                "now_ts_ms": int(now_ts_ms),
                "tf_importance": {"5m": int(tfi.get("5m", 1)), "1h": int(tfi.get("1h", 2)), "4h": int(tfi.get("4h", 3))},
                "zones_cfg": {
                    "macro": {"tf": str(macro_cfg.get("tf") or "4h"), "radius_pct": float(macro_r), "padding_pct": float(macro_p)},
                    "context": {"tf": str(context_cfg.get("tf") or "1h"), "radius_pct": float(context_r), "padding_pct": float(context_p)},
                    "execution": {"tf": str(exec_cfg.get("tf") or "5m"), "radius_pct": float(exec_r), "padding_pct": float(exec_p)},
                },
                "keep_top2_5m": bool(keep_top2_5m),
            },
            "zones": zones,
        }

    if float(grid_pct) <= 0:
        raise ValueError("grid_pct must be > 0")

    candidates: list[dict[str, Any]] = []
    for tf, reg in regs_by_tf.items():
        tf_ms = _tf_ms_from_registry(reg, str(tf))
        imp = int(tfi.get(str(tf), 1))

        for zid in reg.zones.keys():
            ev = zone_representative_event(reg, str(zid), prefer="last")
            if not ev:
                continue
            try:
                level = float(ev["level"])
            except Exception:
                continue
            if level <= 0:
                continue

            role_now = role_from_price(float(level), current_price=float(current_price))
            if role_now not in {"support", "resistance"}:
                continue

            dt_ms = ev.get("dt_ms")
            if not isinstance(dt_ms, int):
                continue
            bars_ago = int(max(0, (int(now_ts_ms) - int(dt_ms)) // int(tf_ms)))

            instant = ev.get("instant") or {}
            tags = instant.get("tags_ccis") or []
            tags_norm = [str(x).strip().lower() for x in tags if str(x).strip()]
            cci_weight = int(len(set(tags_norm)))

            k = grid_key_for_level(level=float(level), anchor_price=float(current_price), grid_pct=float(grid_pct))
            center = grid_level_from_key(key=int(k), anchor_price=float(current_price), grid_pct=float(grid_pct))
            d_to_center = float(abs(float(level) / float(center) - 1.0))

            candidates.append(
                {
                    "tf": str(tf),
                    "tf_importance": int(imp),
                    "role": str(role_now),
                    "grid_key": int(k),
                    "grid_level": float(center),
                    "grid_d_abs": float(d_to_center),
                    "zone_id": str(zid),
                    "event_id": str(ev.get("event_id") or ""),
                    "dt_ms": int(dt_ms),
                    "dt": str(ev.get("dt") or ""),
                    "bars_ago": int(bars_ago),
                    "level": float(level),
                    "cci_tags": tags_norm,
                    "cci_weight": int(cci_weight),
                }
            )

    cells: dict[tuple[str, int], list[dict[str, Any]]] = {}
    for c in candidates:
        if float(c.get("grid_d_abs") or 0.0) > float(grid_pct):
            continue
        key = (str(c["role"]), int(c["grid_key"]))
        cells.setdefault(key, []).append(c)

    out_cells: list[dict[str, Any]] = []
    for (role, gk), arr in cells.items():
        present = {str(x["tf"]) for x in arr}
        if not {"5m", "1h", "4h"}.issubset(present):
            continue

        by_tf: dict[str, list[dict[str, Any]]] = {"5m": [], "1h": [], "4h": []}
        for x in arr:
            tf = str(x["tf"])
            if tf in by_tf:
                by_tf[tf].append(x)

        for tf in by_tf.keys():
            by_tf[tf].sort(key=lambda r: (int(r["bars_ago"]), -int(r["dt_ms"])))

        top_5m = by_tf["5m"][:2] if keep_top2_5m else by_tf["5m"][:1]
        if not top_5m:
            continue

        picks: list[dict[str, Any]] = []
        for p5 in top_5m:
            def _nearest(base: dict[str, Any], others: list[dict[str, Any]]) -> dict[str, Any] | None:
                if not others:
                    return None
                base_level = float(base["level"])
                eligible = [
                    o
                    for o in others
                    if float(abs(float(o["level"]) / base_level - 1.0)) <= float(grid_pct)
                ]
                if not eligible:
                    return None
                best = min(eligible, key=lambda o: (abs(float(o["level"]) - base_level), int(o["bars_ago"])))
                return dict(best)

            p1 = _nearest(p5, by_tf["1h"])
            p4 = _nearest(p5, by_tf["4h"])
            if p1 is None or p4 is None:
                continue

            local_w = int(p5.get("cci_weight") or 0) + int(p1.get("cci_weight") or 0) + int(p4.get("cci_weight") or 0)
            global_w = int(p5.get("tf_importance") or 0) + int(p1.get("tf_importance") or 0) + int(p4.get("tf_importance") or 0)

            picks.append(
                {
                    "members": {
                        "5m": dict(p5),
                        "1h": dict(p1),
                        "4h": dict(p4),
                    },
                    "score": {
                        "importance_global": int(global_w),
                        "importance_local": int(local_w),
                    },
                }
            )

        if not picks:
            continue

        out_cells.append(
            {
                "role": str(role),
                "grid_key": int(gk),
                "grid_level": float(grid_level_from_key(key=int(gk), anchor_price=float(current_price), grid_pct=float(grid_pct))),
                "grid_pct": float(grid_pct),
                "picks": picks,
            }
        )

    out_cells.sort(
        key=lambda c: (
            str(c["role"]),
            min(int(p["members"]["5m"]["bars_ago"]) for p in c["picks"]),
        )
    )

    return {
        "meta": {
            "symbol": str(symbol),
            "grid_pct": float(grid_pct),
            "mode": "grid",
            "current_price": float(current_price),
            "now_ts_ms": int(now_ts_ms),
            "tfs": ["5m", "1h", "4h"],
            "tf_importance": {"5m": int(tfi.get("5m", 1)), "1h": int(tfi.get("1h", 2)), "4h": int(tfi.get("4h", 3))},
            "keep_top2_5m": bool(keep_top2_5m),
        },
        "cells": out_cells,
    }


def extract_execution_pivot_price_weight_table(
    payload: dict[str, Any],
    *,
    current_price: float,
) -> list[dict[str, Any]]:
    mm = str((payload.get("meta") or {}).get("mode") or "").strip().lower()
    out: dict[tuple[str, float], dict[str, Any]] = {}

    def _add_pick(pick: dict[str, Any]) -> None:
        members = pick.get("members") or {}
        p5 = members.get("5m") or {}
        try:
            price = float(p5.get("level"))
        except Exception:
            return
        if price <= 0:
            return
        role = role_from_price(float(price), current_price=float(current_price))
        if role not in {"support", "resistance"}:
            return

        zone_id = str(p5.get("zone_id") or "")
        event_id = str(p5.get("event_id") or "")

        score = pick.get("score") or {}
        g = int(score.get("importance_global") or 0)
        l = int(score.get("importance_local") or 0)
        weight = int(g + l)

        key = (str(role), float(price))
        prev = out.get(key)
        if prev is None or int(prev.get("weight") or 0) < int(weight):
            out[key] = {
                "role": str(role),
                "price": float(price),
                "weight": int(weight),
                "zone_id": str(zone_id),
                "event_id": str(event_id),
            }

    if mm == "zones":
        for mz in payload.get("zones") or []:
            for cz in (mz.get("subzones") or []):
                for ez in (cz.get("subzones") or []):
                    for pick in (ez.get("picks") or []):
                        if isinstance(pick, dict):
                            _add_pick(pick)
    else:
        for cell in payload.get("cells") or []:
            for pick in (cell.get("picks") or []):
                if isinstance(pick, dict):
                    _add_pick(pick)

    items = list(out.values())
    items.sort(key=lambda r: float(r.get("price") or 0.0), reverse=True)
    return items


def is_price_bracketed_by_min_levels(
    price_weight_table: list[dict[str, Any]] | None,
    *,
    current_price: float,
    min_supports: int = 2,
    min_resistances: int = 2,
) -> bool:
    arr = list(price_weight_table or [])

    supports = 0
    resistances = 0
    for r in arr:
        try:
            p = float(r.get("price"))
        except Exception:
            continue
        if p <= 0:
            continue
        role_now = role_from_price(float(p), current_price=float(current_price))
        if role_now == "support":
            supports += 1
        elif role_now == "resistance":
            resistances += 1

    return supports >= int(min_supports) and resistances >= int(min_resistances)


def get_or_refresh_execution_pivot_table(
    *,
    symbol: str,
    current_price: float,
    now_ts_ms: int,
    regs_by_tf: dict[str, PivotRegistry],
    grid_pct: float,
    prev_table: list[dict[str, Any]] | None,
    min_supports: int = 2,
    min_resistances: int = 2,
    keep_top2_5m: bool = True,
    mode: str = "grid",
    zones_cfg: dict[str, dict[str, Any]] | None = None,
) -> dict[str, Any]:
    prev = list(prev_table or [])

    if prev and is_price_bracketed_by_min_levels(
        prev,
        current_price=float(current_price),
        min_supports=int(min_supports),
        min_resistances=int(min_resistances),
    ):
        return {"table": prev, "prev_table": prev, "recomputed": False}

    payload = build_grid_confluence(
        symbol=str(symbol),
        current_price=float(current_price),
        now_ts_ms=int(now_ts_ms),
        grid_pct=float(grid_pct),
        regs_by_tf=regs_by_tf,
        keep_top2_5m=bool(keep_top2_5m),
        mode=str(mode),
        zones_cfg=zones_cfg,
    )
    table = extract_execution_pivot_price_weight_table(payload, current_price=float(current_price))
    return {"table": table, "prev_table": prev, "recomputed": True, "grid_payload": payload}
