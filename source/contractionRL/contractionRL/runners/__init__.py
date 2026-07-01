"""Unified runner for contractionRL: routes PPO/SAC → skrl, contraction → mjrl."""

from contractionRL.runners.contraction_runner import ContractionRunner

__all__ = ["ContractionRunner"]
