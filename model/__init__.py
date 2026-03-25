"""Compatibility package for model namespace.

This package re-exports the active model implementation from
`lprnet.model` so imports using `model.*` remain stable.
"""

from lprnet.model.lprnet import LPRNet, build_lprnet, small_basic_block

__all__ = ["LPRNet", "build_lprnet", "small_basic_block"]