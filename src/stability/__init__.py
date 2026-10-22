"""Standalone Phase 1 analysis; see docs/phase1_stability.md for the contract."""
from .assignment import Assignment
from .margin import StabilityResult, analyze_stability

__all__ = ["Assignment", "StabilityResult", "analyze_stability"]
