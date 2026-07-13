"""Unified runner for contractionRL: routes PPO/SAC → skrl Runner, contraction algorithms (C3M/LQR/SDLQR/C2RL) → native skrl Agent subclasses."""

from contractionRL.runners.contraction_runner import ContractionRunner

__all__ = ["ContractionRunner"]
