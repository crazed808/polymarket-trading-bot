"""
Strategies - Trading Strategy Implementations

This package contains trading strategy implementations:

- base: Base class for all strategies
- flash_crash: Flash crash volatility strategy
- smart_longshot: Optimized late-game longshot strategy (45-120s window)
- momentum: Momentum + orderbook confirmation strategy
- breakout: Consolidation breakout strategy

Usage:
    from strategies.base import BaseStrategy, StrategyConfig
    from strategies.smart_longshot import SmartLongshotStrategy, SmartLongshotConfig
    from strategies.momentum import MomentumStrategy, MomentumConfig
"""

from strategies.base import BaseStrategy, StrategyConfig
from strategies.flash_crash import FlashCrashStrategy, FlashCrashConfig
from strategies.smart_longshot import SmartLongshotStrategy, SmartLongshotConfig
from strategies.momentum import MomentumStrategy, MomentumConfig
from strategies.breakout import BreakoutStrategy, BreakoutConfig

__all__ = [
    "BaseStrategy",
    "StrategyConfig",
    "FlashCrashStrategy",
    "FlashCrashConfig",
    "SmartLongshotStrategy",
    "SmartLongshotConfig",
    "MomentumStrategy",
    "MomentumConfig",
    "BreakoutStrategy",
    "BreakoutConfig",
]
