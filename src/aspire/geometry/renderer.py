"""Rendering ASPIRE training geometry as vector images.

Converts training trajectories into visual representations:
1. TrajectoryImage: 2D embedding of the path through state space
2. VectorFieldImage: Arrows showing direction and magnitude of learning
3. HeatmapImage: Layer-wise evolution over time

These visualizations make the "conscience formation" process observable.
"""

from dataclasses import dataclass, field
from typing import List, Optional, Tuple, Literal
import numpy as np

from .trajectory import TrainingTrajectory
from .state import StateVector


@dataclass
class TrajectoryImage:
    """2D embedding of the training trajectory.

    X/Y coordinates come from dimensionality reduction (PCA, UMAP, etc.)
    Color encodes training progress or accuracy.
    """
    # Coordinates (N x 2)
    coordinates: np.ndarray

    # Color values (N,) - typically cycle index or accuracy
    colors: np.ndarray

    # Arrow vectors for showing direction (N-1 x 2)
    arrows: Optional[np.ndarray] = None

    # Metadata
    method: str = "pca"
    explained_variance: Optional[Tuple[float, float]] = None

    @property
    def x(self) -> np.ndarray:
        return self.coordinates[:, 0]

    @property
    def y(self) -> np.ndarray:
        return self.coordinates[:, 1]


@dataclass
class VectorFieldImage:
    """Vector field showing learning dynamics.

    Each point has a position (from state) and a vector (direction of change).
    This reveals the "flow" of training in state space.
    """
    # Grid positions (M x N x 2)
    positions: np.ndarray

    # Vector magnitudes (M x N x 2) - dx, dy
    vectors: np.ndarray

    # Scalar field (M x N) - e.g., curvature or step size
    scalar_field: Optional[np.ndarray] = None

    # Grid bounds
    x_range: Tuple[float, float] = (0.0, 1.0)
    y_range: Tuple[float, float] = (0.0, 1.0)


@dataclass
class HeatmapImage:
    """Heatmap showing evolution of scalar dimensions over time.

    Rows = dimensions (or dimension groups)
    Columns = training time
    Intensity = scalar value
    """
    # Data matrix (D x T)
    data: np.ndarray

    # Row labels (dimension names)
    row_labels: List[str]

    # Column positions (cycle numbers)
    col_positions: np.ndarray

    # Normalization
    vmin: float = 0.0
    vmax: float = 1.0


class GeometryRenderer:
    """Renders training trajectories as vector images.

    Provides multiple visualization methods:
    - PCA: Linear projection preserving global variance
    - UMAP: Nonlinear embedding preserving local structure
    - t-SNE: Nonlinear embedding emphasizing clusters

    Also generates vector fields and heatmaps.
    """

    def __init__(self, random_state: int = 42):
        self.random_state = random_state

    def render_trajectory_pca(
        self,
        trajectory: TrainingTrajectory,
        color_by: Literal["cycle", "accuracy", "curvature"] = "accuracy",
    ) -> TrajectoryImage:
        """Render trajectory using PCA projection.

        PCA preserves global variance structure, making it ideal for
        seeing the overall "shape" of training.
        """
        vectors = trajectory.vectors
        if len(vectors) < 2:
            return TrajectoryImage(
                coordinates=np.zeros((1, 2)),
                colors=np.array([0.0]),
                method="pca",
            )

        # Center data
        mean = vectors.mean(axis=0)
        centered = vectors - mean

        # Compute covariance and eigenvectors
        cov = np.cov(centered.T)
        eigenvalues, eigenvectors = np.linalg.eigh(cov)

        # Sort by eigenvalue (descending)
        idx = np.argsort(eigenvalues)[::-1]
        eigenvalues = eigenvalues[idx]
        eigenvectors = eigenvectors[:, idx]

        # Project to 2D
        projection_matrix = eigenvectors[:, :2]
        coordinates = centered @ projection_matrix

        # Explained variance
        total_var = eigenvalues.sum()
        if total_var > 0:
            explained = (eigenvalues[0] / total_var, eigenvalues[1] / total_var)
        else:
            explained = (0.0, 0.0)

        # Compute colors
        if color_by == "cycle":
            colors = np.arange(len(trajectory.snapshots), dtype=np.float32)
            colors /= colors.max() + 1e-6
        elif color_by == "accuracy":
            colors = np.array([s.state.accuracy for s in trajectory.snapshots])
        elif color_by == "curvature":
            curvatures = trajectory.compute_curvature()
            colors = np.zeros(len(trajectory.snapshots))
            colors[1:-1] = curvatures
        else:
            colors = np.arange(len(trajectory.snapshots), dtype=np.float32)

        # Compute arrows
        arrows = np.diff(coordinates, axis=0)

        return TrajectoryImage(
            coordinates=coordinates,
            colors=colors,
            arrows=arrows,
            method="pca",
            explained_variance=explained,
        )

    def render_trajectory_umap(
        self,
        trajectory: TrainingTrajectory,
        color_by: Literal["cycle", "accuracy", "curvature"] = "accuracy",
        n_neighbors: int = 15,
        min_dist: float = 0.1,
    ) -> TrajectoryImage:
        """Render trajectory using UMAP projection.

        UMAP preserves local neighborhood structure, revealing
        clusters and transitions in the training path.

        Requires umap-learn package.
        """
        vectors = trajectory.vectors
        if len(vectors) < 2:
            return TrajectoryImage(
                coordinates=np.zeros((1, 2)),
                colors=np.array([0.0]),
                method="umap",
            )

        try:
            import umap
        except ImportError:
            # Fall back to PCA
            print("UMAP not available, falling back to PCA")
            result = self.render_trajectory_pca(trajectory, color_by)
            result.method = "pca (umap fallback)"
            return result

        # Fit UMAP
        reducer = umap.UMAP(
            n_neighbors=min(n_neighbors, len(vectors) - 1),
            min_dist=min_dist,
            random_state=self.random_state,
        )
        coordinates = reducer.fit_transform(vectors)

        # Compute colors
        if color_by == "cycle":
            colors = np.arange(len(trajectory.snapshots), dtype=np.float32)
            colors /= colors.max() + 1e-6
        elif color_by == "accuracy":
            colors = np.array([s.state.accuracy for s in trajectory.snapshots])
        elif color_by == "curvature":
            curvatures = trajectory.compute_curvature()
            colors = np.zeros(len(trajectory.snapshots))
            colors[1:-1] = curvatures
        else:
            colors = np.arange(len(trajectory.snapshots), dtype=np.float32)

        arrows = np.diff(coordinates, axis=0)

        return TrajectoryImage(
            coordinates=coordinates,
            colors=colors,
            arrows=arrows,
            method="umap",
        )

    def render_vector_field(
        self,
        trajectory: TrainingTrajectory,
        grid_size: int = 20,
    ) -> VectorFieldImage:
        """Render learning dynamics as a vector field.

        Creates a grid over the 2D embedding and computes
        the average "flow" direction at each grid cell.
        """
        # First get 2D coordinates
        traj_image = self.render_trajectory_pca(trajectory)
        coords = traj_image.coordinates
        arrows = traj_image.arrows

        if arrows is None or len(arrows) == 0:
            return VectorFieldImage(
                positions=np.zeros((1, 1, 2)),
                vectors=np.zeros((1, 1, 2)),
                x_range=(0, 1),
                y_range=(0, 1),
            )

        # Determine bounds
        x_min, x_max = coords[:, 0].min(), coords[:, 0].max()
        y_min, y_max = coords[:, 1].min(), coords[:, 1].max()

        # Add padding
        x_pad = (x_max - x_min) * 0.1
        y_pad = (y_max - y_min) * 0.1
        x_min -= x_pad
        x_max += x_pad
        y_min -= y_pad
        y_max += y_pad

        # Create grid
        x_edges = np.linspace(x_min, x_max, grid_size + 1)
        y_edges = np.linspace(y_min, y_max, grid_size + 1)

        # Compute grid centers
        x_centers = (x_edges[:-1] + x_edges[1:]) / 2
        y_centers = (y_edges[:-1] + y_edges[1:]) / 2

        positions = np.zeros((grid_size, grid_size, 2))
        vectors = np.zeros((grid_size, grid_size, 2))
        counts = np.zeros((grid_size, grid_size))

        # Assign arrows to grid cells
        for i in range(len(arrows)):
            x, y = coords[i]
            dx, dy = arrows[i]

            # Find grid cell
            xi = np.searchsorted(x_edges[:-1], x) - 1
            yi = np.searchsorted(y_edges[:-1], y) - 1

            xi = max(0, min(grid_size - 1, xi))
            yi = max(0, min(grid_size - 1, yi))

            vectors[yi, xi, 0] += dx
            vectors[yi, xi, 1] += dy
            counts[yi, xi] += 1

        # Average
        mask = counts > 0
        vectors[mask, 0] /= counts[mask]
        vectors[mask, 1] /= counts[mask]

        # Set positions
        for i, y in enumerate(y_centers):
            for j, x in enumerate(x_centers):
                positions[i, j] = [x, y]

        # Compute scalar field (magnitude)
        scalar_field = np.sqrt(vectors[:, :, 0] ** 2 + vectors[:, :, 1] ** 2)

        return VectorFieldImage(
            positions=positions,
            vectors=vectors,
            scalar_field=scalar_field,
            x_range=(x_min, x_max),
            y_range=(y_min, y_max),
        )

    def render_heatmap(
        self,
        trajectory: TrainingTrajectory,
        dimensions: Optional[List[str]] = None,
        subsample: int = 1,
    ) -> HeatmapImage:
        """Render dimension evolution as a heatmap.

        Shows how each scalar dimension evolves over training.
        This reveals which dimensions are "active" during learning.
        """
        from ..core import TokenDimension

        # Default dimensions to show
        if dimensions is None:
            # Token dimension means + critic MAEs
            dimensions = [
                f"{dim.value}_mean" for dim in TokenDimension
            ] + [
                f"{dim.value}_mae" for dim in TokenDimension
            ] + [
                "accuracy",
                "disagreement_mae",
                "revision_rate",
            ]

        # Extract data
        n_snapshots = len(trajectory.snapshots)
        n_dims = len(dimensions)

        data = np.zeros((n_dims, n_snapshots))

        for t, snapshot in enumerate(trajectory.snapshots):
            state = snapshot.state
            for d, dim_name in enumerate(dimensions):
                # Token dimension means
                for token_dim in TokenDimension:
                    if dim_name == f"{token_dim.value}_mean":
                        if token_dim in state.token_stats:
                            data[d, t] = state.token_stats[token_dim].mean
                        else:
                            data[d, t] = 0.5
                    elif dim_name == f"{token_dim.value}_mae":
                        if token_dim in state.token_stats:
                            data[d, t] = state.token_stats[token_dim].prediction_mae
                        else:
                            data[d, t] = 0.5

                # Special dimensions
                if dim_name == "accuracy":
                    data[d, t] = state.accuracy
                elif dim_name == "disagreement_mae":
                    data[d, t] = state.critic_state.disagreement_mae
                elif dim_name == "revision_rate":
                    data[d, t] = state.revision_state.revision_rate

        # Subsample if needed
        if subsample > 1:
            indices = np.arange(0, n_snapshots, subsample)
            data = data[:, indices]
            col_positions = np.array([trajectory.snapshots[i].state.cycle for i in indices])
        else:
            col_positions = np.array([s.state.cycle for s in trajectory.snapshots])

        return HeatmapImage(
            data=data,
            row_labels=dimensions,
            col_positions=col_positions,
            vmin=0.0,
            vmax=1.0,
        )

    def render_before_after(
        self,
        trajectory: TrainingTrajectory,
        early_cycles: int = 20,
        late_cycles: int = 20,
    ) -> dict:
        """Render before/after comparison as a summary.

        Returns coordinates and statistics for visualization.
        """
        comparison = trajectory.get_before_after_comparison(early_cycles, late_cycles)

        # Get 2D embeddings for early and late states
        if len(trajectory.snapshots) < early_cycles + late_cycles:
            return comparison

        traj_image = self.render_trajectory_pca(trajectory)

        early_coords = traj_image.coordinates[:early_cycles]
        late_coords = traj_image.coordinates[-late_cycles:]

        comparison["geometry"] = {
            "early_centroid": early_coords.mean(axis=0).tolist(),
            "late_centroid": late_coords.mean(axis=0).tolist(),
            "early_spread": float(np.std(np.linalg.norm(
                early_coords - early_coords.mean(axis=0), axis=1
            ))),
            "late_spread": float(np.std(np.linalg.norm(
                late_coords - late_coords.mean(axis=0), axis=1
            ))),
            "centroid_distance": float(np.linalg.norm(
                early_coords.mean(axis=0) - late_coords.mean(axis=0)
            )),
        }

        return comparison
