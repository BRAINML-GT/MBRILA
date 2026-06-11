"""Per-region delay models."""

from mbrila.delays.fixed import FixedDelay
from mbrila.delays.none import NoDelay
from mbrila.delays.time_varying import TimeVaryingDelay

__all__ = ["FixedDelay", "NoDelay", "TimeVaryingDelay"]
