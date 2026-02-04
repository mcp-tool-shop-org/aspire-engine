"""Geometry of ASPIRE training - scalar-to-vector visualization.

This module provides tools for visualizing the dimensional evolution
of ASPIRE training as vector images. The core insight is that training
induces dimensional collapse: from uniform, isotropic scalar distributions
to task-aligned, anisotropic structures.

Key concepts:
- StateVector: Snapshot of all relevant scalars at a training checkpoint
- TrainingTrajectory: Sequence of StateVectors over training
- GeometryRenderer: Converts trajectories to vector images

The goal is to make the "conscience formation" process visible.
"""

from .state import StateVector, StateSnapshot
from .trajectory import TrainingTrajectory, TrajectoryMetrics
from .renderer import (
    GeometryRenderer,
    TrajectoryImage,
    VectorFieldImage,
    HeatmapImage,
)

__all__ = [
    "StateVector",
    "StateSnapshot",
    "TrainingTrajectory",
    "TrajectoryMetrics",
    "GeometryRenderer",
    "TrajectoryImage",
    "VectorFieldImage",
    "HeatmapImage",
]
