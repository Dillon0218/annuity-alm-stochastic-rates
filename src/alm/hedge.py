"""Payer swaptions held against the surrender option.

A payer swaption pays when swap rates end above the strike, which is exactly when the
block's credited rate falls behind new money and contract holders leave. Swap rates are
modelled off the Treasury curve (swap spreads are ignored), and the option is valued
at expiry on each path from the Hull-White bond formula and deflated like every other
cash flow, so on average across paths it is worth its premium.
"""
from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np

from .curve import ZeroCurve
from .hull_white import HullWhite, Paths


@dataclass(frozen=True)
class Swaption:
    expiry: float
    tenor: float
    strike: float
    notional: float = 1.0
    kind: str = "payer"

    def price(self, hw: HullWhite) -> float:
        return self.notional * hw.swaption(self.expiry, self.tenor, self.strike, self.kind)

    def value_under(self, hw: HullWhite, curve: ZeroCurve) -> float:
        return self.price(hw.with_curve(curve))

    def pathwise_pv(self, hw: HullWhite, paths: Paths) -> np.ndarray:
        k = int(np.argmin(np.abs(paths.times - self.expiry)))
        if abs(paths.times[k] - self.expiry) > 1e-9:
            raise ValueError("expiry must be on the path grid")
        s, ann = hw.swap_on_path(self.expiry, self.expiry, self.tenor, paths.x[:, k])
        intrinsic = s - self.strike if self.kind == "payer" else self.strike - s
        return self.notional * paths.D[:, k] * ann * np.maximum(intrinsic, 0.0)

    def scaled(self, notional: float) -> "Swaption":
        return replace(self, notional=notional)
