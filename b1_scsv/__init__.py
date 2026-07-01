"""
b1_scsv/__init__.py
===================
Package init for the B1 SCSV sub-module.  Re-exports the primary API so
callers can write:

    from b1_scsv import SCSV, SCORE_ALLOW, SCORE_BLOCK
"""

from .scsv import SCSV, SCORE_ALLOW, SCORE_BLOCK

__all__ = ["SCSV", "SCORE_ALLOW", "SCORE_BLOCK"]
