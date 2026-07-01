"""
b2_csia/__init__.py
===================
Package init for the B2 CSIA sub-module.  Re-exports the primary API so
callers can write:

    from b2_csia import CSIA
"""

from .csia import CSIA

__all__ = ["CSIA"]
