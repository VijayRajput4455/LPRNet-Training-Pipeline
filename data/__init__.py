"""Compatibility package for data namespace.

This package re-exports the active dataset loader from `lprnet.data`
so imports using `data.*` continue to resolve at package level.
"""

from lprnet.data.loader import CHARS, CHARS_DICT, LPRDataLoader

__all__ = ["CHARS", "CHARS_DICT", "LPRDataLoader"]