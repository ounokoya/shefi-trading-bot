from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ExtremeConfluencePreset:
    name: str
    series_cols: list[str]
    cci_fast_threshold: float | None = None
    cci_medium_threshold: float | None = None
    cci_slow_threshold: float | None = None


def get_extreme_confluence_preset(name: str) -> ExtremeConfluencePreset:
    k = str(name).strip().lower().replace(" ", "_")

    # 1) scalping
    if k in {"scalping"}:
        return ExtremeConfluencePreset(
            name="scalping",
            series_cols=["close", "vwma_4", "macd_line", "macd_hist", "cci_30"],
            cci_fast_threshold=100.0,
        )

    # 2) intraday strict
    if k in {"intraday_strict"}:
        return ExtremeConfluencePreset(
            name="intraday_strict",
            series_cols=["close", "vwma_4", "macd_line", "macd_hist", "cci_30", "cci_120"],
            cci_fast_threshold=100.0,
            cci_medium_threshold=100.0,
        )

    # 3) intraday standard
    if k in {"intraday_standard"}:
        return ExtremeConfluencePreset(
            name="intraday_standard",
            series_cols=["close", "vwma_4", "macd_line", "macd_hist", "cci_30", "cci_120"],
            cci_fast_threshold=100.0,
            cci_medium_threshold=0.0,
        )

    # 4) swing strict (vwma medium + cci slow strict)
    if k in {
        "swing_strict",
        "swing_strict_full",
        "swing_strict_vwma12",
        # backward-compatible aliases
        "intraday_strict_plus",
        "intraday_strict_full",
        "intraday_strict_vwma12",
    }:
        return ExtremeConfluencePreset(
            name="swing_strict",
            series_cols=["close", "vwma_4", "vwma_12", "macd_line", "macd_hist", "cci_30", "cci_120", "cci_300"],
            cci_fast_threshold=100.0,
            cci_medium_threshold=100.0,
            cci_slow_threshold=100.0,
        )

    # 5) swing standard (vwma medium + cci slow relaxed)
    if k in {
        "swing_standard",
        "swing_standard_full",
        "swing_standard_vwma12",
        "swing_standard_vwma_medium",
        # backward-compatible aliases
        "intraday_standard_plus",
        "intraday_standard_full",
        "intraday_standard_vwma12",
        "intraday_standard_vwma_medium",
    }:
        return ExtremeConfluencePreset(
            name="swing_standard",
            series_cols=["close", "vwma_4", "vwma_12", "macd_line", "macd_hist", "cci_30", "cci_120", "cci_300"],
            cci_fast_threshold=100.0,
            cci_medium_threshold=100.0,
            cci_slow_threshold=0.0,
        )

    raise ValueError(
        "Unknown preset. Valid presets: scalping, intraday_strict, intraday_standard, "
        "swing_strict (aliases intraday_strict_plus, intraday_strict_vwma12), "
        "swing_standard (aliases intraday_standard_plus, intraday_standard_vwma12)"
    )
