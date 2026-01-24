from __future__ import annotations

import math


def _is_bad(x: float) -> bool:
    return math.isnan(x) or math.isinf(x)
