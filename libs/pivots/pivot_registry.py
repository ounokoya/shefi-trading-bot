from __future__ import annotations

import bisect
import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable

import pandas as pd


def _tf_to_ms(tf: str) -> int:
    s = str(tf).strip().lower()
    if not s:
        raise ValueError("tf cannot be empty")

    if s.endswith("m") and s[:-1].isdigit():
        return int(s[:-1]) * 60_000
    if s.endswith("h") and s[:-1].isdigit():
        return int(s[:-1]) * 3_600_000
    if s.endswith("d") and s[:-1].isdigit():
        return int(s[:-1]) * 86_400_000
    if s.endswith("w") and s[:-1].isdigit():
        return int(s[:-1]) * 7 * 86_400_000

    if s == "d":
        return 86_400_000
    if s == "w":
        return 7 * 86_400_000

    raise ValueError(f"unsupported tf format: {tf}")


def _primary_cat(tags: Iterable[str]) -> str:
    t = {str(x).strip().lower() for x in tags if str(x).strip()}
    if "slow" in t:
        return "slow"
    if "medium" in t:
        return "medium"
    if "fast" in t:
        return "fast"
    return ""


def _finite_float(x: object) -> float | None:
    v = pd.to_numeric(x, errors="coerce")
    if pd.isna(v):
        return None
    f = float(v)
    if not math.isfinite(f):
        return None
    return f


@dataclass
class PivotRegistry:
    events: dict[str, dict[str, Any]] = field(default_factory=dict)
    zones: dict[str, dict[str, Any]] = field(default_factory=dict)
    meta: dict[str, Any] = field(default_factory=dict)
    zone_index: dict[str, dict[int, list[str]]] = field(default_factory=lambda: {"support": {}, "resistance": {}})

    @classmethod
    def empty(
        cls,
        *,
        symbol: str,
        tf: str,
        eps: float,
    ) -> "PivotRegistry":
        out = cls()
        out.meta = {
            "symbol": str(symbol),
            "tf": str(tf),
            "tf_ms": int(_tf_to_ms(tf)),
            "eps": float(eps),
            "last_ts": None,
            "next_zone_id": 1,
        }
        out.zone_index = {"support": {}, "resistance": {}}
        return out

    @classmethod
    def from_json(cls, path: str | Path) -> "PivotRegistry":
        p = Path(path)
        obj = json.loads(p.read_text())
        out = cls(
            events=dict(obj.get("events") or {}),
            zones=dict(obj.get("zones") or {}),
            meta=dict(obj.get("meta") or {}),
        )
        out._normalize_zones_event_ids()
        out._rebuild_zone_index()
        return out

    def to_json(self, path: str | Path) -> None:
        p = Path(path)
        payload = {
            "events": self.events,
            "zones": self.zones,
            "meta": self.meta,
        }
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(payload, indent=2, sort_keys=True))

    @classmethod
    def from_extremes_df(
        cls,
        extremes_df: pd.DataFrame,
        *,
        symbol: str,
        tf: str,
        eps: float,
    ) -> "PivotRegistry":
        reg = cls.empty(symbol=symbol, tf=tf, eps=eps)
        reg.update_from_extremes_df(extremes_df)
        return reg

    def update_from_extremes_df(self, extremes_df: pd.DataFrame) -> None:
        if len(extremes_df) == 0:
            return

        for c in ("ts", "kind", "close", "cci_category"):
            if c not in extremes_df.columns:
                raise ValueError(f"Missing required column in extremes_df: {c}")

        work = extremes_df.copy()
        work["ts"] = pd.to_numeric(work["ts"], errors="coerce").astype("Int64")
        work = work.dropna(subset=["ts", "kind", "close", "cci_category"]).reset_index(drop=True)
        if len(work) == 0:
            return

        if "dt" not in work.columns:
            work["dt"] = pd.to_datetime(work["ts"].astype(int), unit="ms", utc=True).dt.strftime(
                "%Y-%m-%d %H:%M:%S UTC"
            )

        work = work.sort_values(["ts", "kind", "cci_category"]).reset_index(drop=True)

        # New path (preferred): one updatable event per tranche (ongoing tranche supported).
        if "tranche_start_ts" in work.columns:
            work["tranche_start_ts"] = pd.to_numeric(work["tranche_start_ts"], errors="coerce").astype("Int64")
            work = work.dropna(subset=["tranche_start_ts"]).reset_index(drop=True)
            if len(work) == 0:
                return

            work = work.sort_values(["tranche_start_ts", "kind", "ts", "cci_category"]).reset_index(drop=True)
            for (tranche_start_ts_i, kind), grp in work.groupby(["tranche_start_ts", "kind"], sort=False):
                tranche_start_ts_ms = int(tranche_start_ts_i)
                kind_s = str(kind).upper()
                if kind_s not in {"LOW", "HIGH"}:
                    continue

                grp2 = grp.sort_values(["ts", "cci_category"]).reset_index(drop=True)
                last_row = grp2.iloc[-1]

                ts_raw = pd.to_numeric(last_row["ts"], errors="coerce")
                if pd.isna(ts_raw):
                    continue
                ts_ms = int(ts_raw)

                level = _finite_float(last_row["close"])
                if level is None:
                    continue

                dt = str(last_row["dt"])

                tags = {str(x).strip().lower() for x in grp["cci_category"].tolist() if str(x).strip()}
                tags = {t for t in tags if t in {"fast", "medium", "slow"}}
                if not tags:
                    continue

                role = "support" if kind_s == "LOW" else "resistance"
                cat = _primary_cat(tags)
                if not cat:
                    continue

                s_fast = None
                s_medium = None
                s_slow = None
                if "cci_value" in grp.columns:
                    if "fast" in tags:
                        s_fast = _finite_float(grp[grp["cci_category"].astype(str) == "fast"]["cci_value"].iloc[0])
                    if "medium" in tags:
                        s_medium = _finite_float(grp[grp["cci_category"].astype(str) == "medium"]["cci_value"].iloc[0])
                    if "slow" in tags:
                        s_slow = _finite_float(grp[grp["cci_category"].astype(str) == "slow"]["cci_value"].iloc[0])

                # Stable ID per tranche: allow updating ts/level/tags while tranche is ongoing.
                event_id = f"{tranche_start_ts_ms}:{role}:{kind_s}"

                ev = {
                    "event_id": str(event_id),
                    "dt_ms": int(ts_ms),
                    "dt": str(dt),
                    "level": float(level),
                    "cat": str(cat),
                    "role": str(role),
                    "kind": str(kind_s),
                    "tranche_start_ts": int(tranche_start_ts_ms),
                    "tranche": {
                        "tags_ccis": sorted(tags),
                        "strength_fast": s_fast,
                        "strength_medium": s_medium,
                        "strength_slow": s_slow,
                    },
                    # Backward compat: some callers still read `instant.*`
                    "instant": {
                        "tags_ccis": sorted(tags),
                        "strength_fast": s_fast,
                        "strength_medium": s_medium,
                        "strength_slow": s_slow,
                    },
                }

                self._upsert_event(ev)

            return

        # Legacy path: dt-based immutable events.
        for (ts_i, kind), grp in work.groupby(["ts", "kind"], sort=False):
            ts_ms = int(ts_i)
            kind_s = str(kind).upper()
            if kind_s not in {"LOW", "HIGH"}:
                continue

            level = _finite_float(grp["close"].iloc[0])
            if level is None:
                continue

            dt = str(grp["dt"].iloc[0])

            tags = {str(x).strip().lower() for x in grp["cci_category"].tolist() if str(x).strip()}
            tags = {t for t in tags if t in {"fast", "medium", "slow"}}
            if not tags:
                continue

            role = "support" if kind_s == "LOW" else "resistance"
            cat = _primary_cat(tags)
            if not cat:
                continue

            s_fast = None
            s_medium = None
            s_slow = None
            if "cci_value" in grp.columns:
                if "fast" in tags:
                    s_fast = _finite_float(grp[grp["cci_category"].astype(str) == "fast"]["cci_value"].iloc[0])
                if "medium" in tags:
                    s_medium = _finite_float(grp[grp["cci_category"].astype(str) == "medium"]["cci_value"].iloc[0])
                if "slow" in tags:
                    s_slow = _finite_float(grp[grp["cci_category"].astype(str) == "slow"]["cci_value"].iloc[0])

            event_id = f"{ts_ms}:{cat}:{role}:{kind_s}"
            if event_id in self.events:
                continue

            ev = {
                "event_id": str(event_id),
                "dt_ms": int(ts_ms),
                "dt": str(dt),
                "level": float(level),
                "cat": str(cat),
                "role": str(role),
                "kind": str(kind_s),
                "instant": {
                    "tags_ccis": sorted(tags),
                    "strength_fast": s_fast,
                    "strength_medium": s_medium,
                    "strength_slow": s_slow,
                },
            }

            self._insert_event(ev)

    def iter_events(self) -> Iterable[dict[str, Any]]:
        return self.events.values()

    def pick_nearest_level(
        self,
        *,
        current_ts: int,
        current_price: float,
        category: str,
        role: str,
        kind_filter: str,
        threshold_pct: float,
        threshold_mode: str,
    ) -> dict[str, Any] | None:
        cat_req = str(category).strip().lower()
        if cat_req not in {"fast", "medium", "slow"}:
            raise ValueError(f"unknown category: {category}")

        rr = str(role).strip().lower()
        if rr not in {"support", "resistance"}:
            raise ValueError(f"unknown role: {role}")

        kf = str(kind_filter).strip().lower()
        if kf not in {"high", "low", "both"}:
            raise ValueError(f"unknown kind_filter: {kind_filter}")

        thr = float(threshold_pct)
        if thr < 0:
            raise ValueError("threshold_pct must be >= 0")
        tm = str(threshold_mode).strip().lower()
        if tm not in {"exclusive", "inclusive"}:
            raise ValueError(f"unknown threshold_mode: {threshold_mode}")

        tf_ms = int(self.meta.get("tf_ms") or _tf_to_ms(str(self.meta.get("tf") or "")))
        eps = float(self.meta.get("eps") or 0.0)

        best: tuple[int, float, int, dict[str, Any]] | None = None

        for ev in self.iter_events():
            confluence = ev.get("tranche") or ev.get("instant") or {}
            tags = confluence.get("tags_ccis") or []
            if cat_req not in {str(x).strip().lower() for x in tags}:
                continue

            kind = str(ev.get("kind")).strip().upper()
            if kf == "high" and kind != "HIGH":
                continue
            if kf == "low" and kind != "LOW":
                continue

            level = _finite_float(ev.get("level"))
            if level is None or current_price == 0 or not math.isfinite(float(current_price)):
                continue

            d_pct = float((level - float(current_price)) / float(current_price))
            if rr == "resistance" and not (d_pct > 0):
                continue
            if rr == "support" and not (d_pct < 0):
                continue

            d_abs = abs(d_pct)
            if thr > 0:
                if tm == "exclusive" and d_abs < thr:
                    continue
                if tm == "inclusive" and d_abs > thr:
                    continue

            dt_ms = int(ev.get("dt_ms"))
            if tf_ms <= 0:
                continue
            bars_ago = int((int(current_ts) - dt_ms) // tf_ms)
            if bars_ago < 0:
                continue

            score = (bars_ago, d_abs, -dt_ms)
            if best is None or score < best[:3]:
                out = dict(ev)
                out["bars_ago"] = int(bars_ago)
                out["d_pct"] = float(d_pct)
                out["d_abs"] = float(d_abs)
                out["d_price"] = float(level - float(current_price))
                out["current_price"] = float(current_price)
                out["eps"] = float(eps)
                best = (int(bars_ago), float(d_abs), int(-dt_ms), out)

        return None if best is None else best[3]

    def temporal_memory_solidity(
        self,
        *,
        current_ts: int,
        current_price: float,
        side: str,
        radius_pct: float,
        exclude_tranche_start_ts: int | None = None,
        min_fast: int = 4,
        min_medium: int = 2,
        min_slow: int = 1,
        max_events: int = 50,
    ) -> dict[str, Any]:
        s = str(side).strip().upper()
        if s not in {"LONG", "SHORT"}:
            raise ValueError(f"Unexpected side: {side}")

        r = float(radius_pct)
        if r < 0:
            raise ValueError("radius_pct must be >= 0")

        cp = _finite_float(current_price)
        if cp is None or float(cp) == 0.0:
            return {
                "is_solid": False,
                "side": str(s),
                "kind": ("LOW" if s == "LONG" else "HIGH"),
                "radius_pct": float(r),
                "n": 0,
                "n_fast": 0,
                "n_medium": 0,
                "n_slow": 0,
                "score": 0,
                "events": [],
            }

        kind_req = "LOW" if s == "LONG" else "HIGH"
        tf_ms = int(self.meta.get("tf_ms") or 0)

        selected: list[dict[str, Any]] = []
        for ev in self.iter_events():
            if exclude_tranche_start_ts is not None:
                ts0 = ev.get("tranche_start_ts")
                try:
                    ts0_i = int(ts0) if ts0 is not None else None
                except Exception:
                    ts0_i = None
                if ts0_i is not None and int(ts0_i) == int(exclude_tranche_start_ts):
                    continue

            dt_ms_raw = ev.get("dt_ms")
            if not isinstance(dt_ms_raw, int):
                try:
                    dt_ms = int(dt_ms_raw)
                except Exception:
                    continue
            else:
                dt_ms = int(dt_ms_raw)

            if int(dt_ms) >= int(current_ts):
                continue

            kind = str(ev.get("kind") or "").strip().upper()
            if kind != str(kind_req):
                continue

            lvl = _finite_float(ev.get("level"))
            if lvl is None:
                continue

            d_pct = float((float(lvl) - float(cp)) / float(cp))
            d_abs = abs(float(d_pct))
            if float(d_abs) > float(r):
                continue

            confluence = ev.get("tranche") or ev.get("instant") or {}
            tags = confluence.get("tags_ccis") or []
            cat = _primary_cat(tags)
            if cat not in {"fast", "medium", "slow"}:
                continue

            bars_ago: int | None = None
            if tf_ms > 0:
                bars_ago2 = int((int(current_ts) - int(dt_ms)) // tf_ms)
                if bars_ago2 < 0:
                    continue
                bars_ago = int(bars_ago2)

            selected.append(
                {
                    "event_id": str(ev.get("event_id") or ""),
                    "dt_ms": int(dt_ms),
                    "dt": ev.get("dt"),
                    "level": float(lvl),
                    "role": ev.get("role"),
                    "kind": str(kind),
                    "zone_id": ev.get("zone_id"),
                    "primary_cat": str(cat),
                    "tags_ccis": list(tags),
                    "bars_ago": bars_ago,
                    "d_pct": float(d_pct),
                    "d_abs": float(d_abs),
                    "d_price": float(float(lvl) - float(cp)),
                }
            )

        selected.sort(key=lambda x: int(x.get("dt_ms") or 0), reverse=True)
        if int(max_events) > 0 and len(selected) > int(max_events):
            selected = selected[: int(max_events)]

        n_fast = 0
        n_medium = 0
        n_slow = 0
        score = 0
        for r_ev in selected:
            c = str(r_ev.get("primary_cat") or "").strip().lower()
            if c == "slow":
                n_slow += 1
                score += 3
            elif c == "medium":
                n_medium += 1
                score += 2
            elif c == "fast":
                n_fast += 1
                score += 1

        is_solid = bool((n_slow >= int(min_slow)) or (n_medium >= int(min_medium)) or (n_fast >= int(min_fast)))
        return {
            "is_solid": bool(is_solid),
            "side": str(s),
            "kind": str(kind_req),
            "radius_pct": float(r),
            "n": int(len(selected)),
            "n_fast": int(n_fast),
            "n_medium": int(n_medium),
            "n_slow": int(n_slow),
            "score": int(score),
            "events": list(selected),
        }

    def _pct_dist(self, a: float, b: float) -> float:
        if b == 0:
            return float("inf")
        return float(abs((a / b) - 1.0))

    def _bucket_key(self, level: float) -> int:
        eps = float(self.meta.get("eps") or 0.0)
        base = math.log1p(eps) if eps > 0 else 0.0
        if base <= 0 or level <= 0 or (not math.isfinite(level)):
            return 0
        return int(math.floor(math.log(level) / base))

    def _zone_ref_level(self, zone_id: str) -> float | None:
        z = self.zones.get(str(zone_id))
        if not z:
            return None
        sel = z.get("selected_event_id")
        if sel is not None and str(sel) in self.events:
            return _finite_float(self.events[str(sel)].get("level"))
        ids = z.get("event_ids") or []
        if ids:
            eid = str(ids[0])
            if eid in self.events:
                return _finite_float(self.events[eid].get("level"))
        return None

    def _find_zone_id(self, *, role: str, level: float) -> str | None:
        rr = str(role).strip().lower()
        eps = float(self.meta.get("eps") or 0.0)
        if eps <= 0:
            return None

        b = self._bucket_key(float(level))
        candidates: list[str] = []
        for kk in range(b - 1, b + 2):
            candidates.extend(self.zone_index.get(rr, {}).get(int(kk), []))

        best: tuple[float, str] | None = None
        for zid in candidates:
            ref = self._zone_ref_level(str(zid))
            if ref is None:
                continue
            d = self._pct_dist(float(level), float(ref))
            if d <= eps:
                if best is None or d < best[0]:
                    best = (float(d), str(zid))

        return None if best is None else best[1]

    def _new_zone_id(self) -> str:
        n = int(self.meta.get("next_zone_id") or 1)
        self.meta["next_zone_id"] = int(n + 1)
        return f"z{n}"

    def _insert_event(self, ev: dict[str, Any]) -> None:
        event_id = str(ev["event_id"])
        if event_id in self.events:
            return

        role = str(ev.get("role") or "").strip().lower()
        if role not in {"support", "resistance"}:
            return

        level = _finite_float(ev.get("level"))
        if level is None:
            return

        zid = self._find_zone_id(role=role, level=float(level))
        if zid is None:
            zid = self._new_zone_id()
            self.zones[zid] = {
                "role": str(role),
                "eps": float(self.meta.get("eps") or 0.0),
                "event_ids": [],
                "selected_event_id": event_id,
            }

        ev2 = dict(ev)
        ev2["zone_id"] = str(zid)

        self.events[event_id] = ev2
        z = self.zones[str(zid)]
        ids = z.get("event_ids")
        if not isinstance(ids, list):
            ids = []
            z["event_ids"] = ids

        ts_ms = int(ev2.get("dt_ms"))
        if ids:
            dts = [int(self.events[str(x)]["dt_ms"]) for x in ids if str(x) in self.events]
            pos = bisect.bisect_right(dts, ts_ms)
            ids.insert(int(pos), event_id)
        else:
            ids.append(event_id)

        b = self._bucket_key(float(level))
        idx = self.zone_index.setdefault(role, {})
        arr = idx.setdefault(int(b), [])
        if str(zid) not in arr:
            arr.append(str(zid))

        last = self.meta.get("last_ts")
        if last is None or int(ts_ms) > int(last):
            self.meta["last_ts"] = int(ts_ms)

    def _upsert_event(self, ev: dict[str, Any]) -> None:
        event_id = str(ev.get("event_id") or "")
        if not event_id:
            return

        if event_id not in self.events:
            self._insert_event(ev)
            return

        existing = dict(self.events.get(event_id) or {})
        old_zone_id = existing.get("zone_id")

        role = str(ev.get("role") or "").strip().lower()
        if role not in {"support", "resistance"}:
            return

        level = _finite_float(ev.get("level"))
        if level is None:
            return

        zid = self._find_zone_id(role=role, level=float(level))
        if zid is None:
            zid = self._new_zone_id()
            self.zones[zid] = {
                "role": str(role),
                "eps": float(self.meta.get("eps") or 0.0),
                "event_ids": [],
                "selected_event_id": event_id,
            }

        # Remove from old zone (if any)
        if old_zone_id is not None and str(old_zone_id) in self.zones and str(old_zone_id) != str(zid):
            zold = self.zones.get(str(old_zone_id))
            if zold is not None:
                ids_old = zold.get("event_ids")
                if isinstance(ids_old, list) and event_id in ids_old:
                    ids_old[:] = [x for x in ids_old if str(x) != event_id]
                if str(zold.get("selected_event_id")) == event_id:
                    zold["selected_event_id"] = ids_old[0] if isinstance(ids_old, list) and ids_old else None
                if not (isinstance(ids_old, list) and ids_old):
                    del self.zones[str(old_zone_id)]

        # Update event payload (zone_id may change)
        ev2 = dict(ev)
        ev2["zone_id"] = str(zid)
        self.events[event_id] = ev2

        # Re-insert into target zone sorted by dt
        z = self.zones.get(str(zid))
        if z is None:
            return
        ids = z.get("event_ids")
        if not isinstance(ids, list):
            ids = []
            z["event_ids"] = ids
        if event_id in ids:
            ids[:] = [x for x in ids if str(x) != event_id]

        ts_ms = int(ev2.get("dt_ms"))
        if ids:
            dts = [int(self.events[str(x)]["dt_ms"]) for x in ids if str(x) in self.events]
            pos = bisect.bisect_right(dts, ts_ms)
            ids.insert(int(pos), event_id)
        else:
            ids.append(event_id)

        # Keep selected_event_id valid
        sel = z.get("selected_event_id")
        if sel is None or str(sel) not in self.events:
            z["selected_event_id"] = event_id

        # Zone index might be stale if the event moved.
        self._rebuild_zone_index()

        last = self.meta.get("last_ts")
        if last is None or int(ts_ms) > int(last):
            self.meta["last_ts"] = int(ts_ms)

    def _normalize_zones_event_ids(self) -> None:
        for zid, z in self.zones.items():
            ids = z.get("event_ids")
            if not isinstance(ids, list) or not ids:
                continue

            pairs: list[tuple[int, str]] = []
            for eid in ids:
                k = str(eid)
                ev = self.events.get(k)
                if not ev:
                    continue
                dt_ms = ev.get("dt_ms")
                if not isinstance(dt_ms, int):
                    continue
                pairs.append((int(dt_ms), k))

            pairs.sort(key=lambda x: x[0])
            z["event_ids"] = [k for _, k in pairs]

    def _rebuild_zone_index(self) -> None:
        self.zone_index = {"support": {}, "resistance": {}}
        for zid, z in self.zones.items():
            role = str(z.get("role") or "").strip().lower()
            if role not in {"support", "resistance"}:
                continue
            ref = self._zone_ref_level(str(zid))
            if ref is None:
                continue
            b = self._bucket_key(float(ref))
            idx = self.zone_index.setdefault(role, {})
            arr = idx.setdefault(int(b), [])
            if str(zid) not in arr:
                arr.append(str(zid))
