"""
HydroBrain — система підтримки рішень щодо поливу (водний баланс FAO-56 + прогноз погоди +
злиття з датчиками + онлайн-навчання на правках користувача).

Автор: Стеценко Ярослава, 2026.
"""
from .brain import HydroBrain
from .params import CATALOG, GROUP_NAMES, input_schema
from .sources import DataSources, SourceError

__version__ = "1.11.2"
__all__ = ["HydroBrain", "DataSources", "SourceError", "CATALOG", "GROUP_NAMES", "input_schema"]
