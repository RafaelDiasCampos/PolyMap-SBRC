from concurrent.futures import ThreadPoolExecutor
from itertools import product
import os
from typing import Callable, Optional
import pandas as pd
import numpy as np
import cvxpy as cp
from scipy.spatial import ConvexHull
from sklearn.cluster import AffinityPropagation
from ..data_preparer import DataPreparer


class PolytopeAttack:
    def __init__(
        self,
        attack_samples: pd.DataFrame,
        batch_size: int,
        random_state: int
    ):
        self.batch_size = batch_size
        self.random_state = random_state

        self.dataPreparer = DataPreparer(attack_samples, batch_size=batch_size,
                                         random_state=self.random_state, scaler_type="minmax")

        self.columns = attack_samples.columns.tolist()[
            :-1]  # Exclude label column

        self.categorical_cols = attack_samples.select_dtypes(
            include=["object", "category", "string"]
        ).columns.tolist()[:-1]  # Exclude label column

    def rebuild_df(self, points, categorical_group, unscale_and_decode=True):
        points_df = pd.DataFrame(
            points, columns=categorical_group["data"].columns)

        for parameter, value in categorical_group["parameters"].items():
            points_df[parameter] = value

        points_df = points_df[self.columns]

        if not unscale_and_decode:
            return points_df

        points_df_decoded = self.dataPreparer.unscale_and_decode(points_df)
        return points_df_decoded

    def build_categorical_groups(self, normal_samples_encoded_df: pd.DataFrame):
        # Get the unique values for each categorical column
        unique_values = [normal_samples_encoded_df[col].unique()
                         for col in self.categorical_cols]

        self.categorical_groups = []

        # Loop through all combinations
        for combo in product(*unique_values):
            row_dict = dict(zip(self.categorical_cols, combo))

            data = normal_samples_encoded_df[
                (normal_samples_encoded_df[self.categorical_cols] == pd.Series(
                    row_dict)).all(axis=1)
            ]

            data = data.drop(columns=row_dict.keys())

            if data.shape[0] == 0:
                continue

            sample_group = {
                "parameters": row_dict,
                "data": data
            }

            self.categorical_groups.append(sample_group)

    def find_clusters(self):
        print("Finding clusters")
        for categorical_group in self.categorical_groups:
            clustering_model = AffinityPropagation(
                random_state=self.random_state)
            group_clusters = clustering_model.fit_predict(
                categorical_group["data"])
            n_clusters = np.array(clustering_model.cluster_centers_).shape[0]

            categorical_group["n_clusters"] = n_clusters
            categorical_group["clusters"] = group_clusters
            categorical_group["centers"] = clustering_model.cluster_centers_

    def query_blackbox(
            self,
            data: pd.DataFrame,
            blackbox_predict: Callable[[pd.DataFrame], np.ndarray],
            training_phase: bool = True) -> np.ndarray:

        blackbox_preds = blackbox_predict(data)

        if training_phase:
            self.query_stats["n_queries"] += np.int64(len(blackbox_preds))
            self.query_stats["benign_queries"] += np.sum(blackbox_preds == 1)
            self.query_stats["malicious_queries"] += np.sum(
                blackbox_preds == 0)

        return blackbox_preds

    def check_centers(self, blackbox_predict: Callable[[pd.DataFrame], np.ndarray]):
        print("Checking cluster centers")
        n_good_centers = 0
        n_total_centers = 0

        for categorical_group in self.categorical_groups:
            if len(categorical_group["centers"]) == 0:
                continue
            centers_df = self.rebuild_df(
                categorical_group["centers"], categorical_group, self.dataPreparer)
            preds = self.query_blackbox(
                centers_df, blackbox_predict, training_phase=False)
            normal_percentage = np.sum(preds)

            n_good_centers += normal_percentage
            n_total_centers += len(centers_df)

        print(
            f"Percentage of centers predicted as normal: {n_good_centers / n_total_centers * 100:.2f}% ({n_good_centers} out of {n_total_centers})")

    def fit(self,
            normal_samples: pd.DataFrame,
            blackbox_predict: Callable[[pd.DataFrame], np.ndarray],
            n_rays: int = 50,
            step_size: float = 0.05
            ):

        self.query_stats = {
            "n_queries": 0,
            "malicious_queries": 0,
            "benign_queries": 0,
        }

        normal_samples_encoded, _ = self.dataPreparer.scale_and_encode(
            normal_samples)

        normal_samples_encoded_df = pd.DataFrame(
            normal_samples_encoded, columns=self.columns)

        # Separate normal samples into groups depending on categorical features
        self.build_categorical_groups(normal_samples_encoded_df)

        # Find clusters in each group
        self.find_clusters()

        # Check how many cluster centers are classified as normal by the blackbox
        self.check_centers(blackbox_predict)

        # Map polytopes for each cluster center
        print("Mapping polytopes")
        for categorical_group in self.categorical_groups:
            # Skip groups with no clusters
            if categorical_group["n_clusters"] == 0:
                continue

            # Convert to DataFrame
            centers_df = self.rebuild_df(
                categorical_group["centers"], categorical_group, unscale_and_decode=False)

            # Drop categorical columns for mapping
            centers_df = centers_df.drop(columns=self.categorical_cols)

            def test_inside(x):
                x_df = self.rebuild_df(
                    x.reshape(1, -1), categorical_group, unscale_and_decode=True)

                preds = self.query_blackbox(
                    x_df, blackbox_predict)

                return preds[0] == 1  # True if predicted as normal

            categorical_group["hulls"] = []

            for start_point in centers_df.values:
                # Skip centers that are not inside the polytope
                if not test_inside(start_point):
                    continue

                # Skip centers that are inside another hull
                is_inside_another_hull = False
                for hull in categorical_group["hulls"]:
                    if self.is_inside_hull(start_point, hull):
                        is_inside_another_hull = True
                        break
                if is_inside_another_hull:
                    continue

                # Map polytope points starting from the cluster center
                boundary_points = self.map_polytope(
                    test_inside,
                    start_point,
                    n_rays=n_rays,
                    step_size=step_size
                )

                # Attempt to compute convex hull from boundary points
                try:
                    hull = self.compute_convex_hull(
                        boundary_points, start_point)
                    categorical_group["hulls"].append(hull)
                except Exception as e:
                    print(
                        f"Could not compute hull for center {start_point}: {e}")

        print("Polytope mapping completed.")

    def find_closest_hull_point(
        self,
        sample: np.ndarray,
        fixed_idx: tuple = (),
        move_inside: float = 0.0
    ) -> np.ndarray:
        closest_points = []

        # Try to find categorical group that matches sample
        group = None
        for categorical_group in self.categorical_groups:
            match = True
            for parameter, value in categorical_group["parameters"].items():
                col_idx = self.columns.index(parameter)
                if sample[col_idx] != value:
                    match = False
                    break
            if match:
                if "hulls" in categorical_group and len(categorical_group["hulls"]) > 0:
                    group = categorical_group
                break

        # If match found, only use that group, else use all groups
        if group is not None:
            groups = [group]
        else:
            groups = self.categorical_groups

        # Drop categorical columns for hull checking
        sample_no_cat = np.delete(
            sample,
            [self.columns.index(col) for col in self.categorical_cols]
        )

        for categorical_group in groups:
            if "hulls" not in categorical_group:
                continue
            for hull_info in categorical_group["hulls"]:
                closest_point_no_cat = self.closest_point_in_hull(
                    hull_info, sample_no_cat, fixed_idx=fixed_idx, move_inside=move_inside)

                # Rebuild full sample with categorical features
                closest_point = self.rebuild_df(
                    closest_point_no_cat.reshape(1, -1),
                    categorical_group,
                    unscale_and_decode=False
                ).values[0]

                closest_points.append((closest_point, hull_info))

        if len(closest_points) == 0:
            raise RuntimeError("No hulls available to find closest point")

        # Find closest point to the original sample
        dists = [np.linalg.norm(sample - p[0]) for p in closest_points]
        best_point = closest_points[np.argmin(dists)][0]

        return best_point

    def generate_samples(
        self,
        samples_df: pd.DataFrame,
        fixed_idx: tuple = (),
        move_inside: float = 0.0,
        n_threads: Optional[int] = None
    ) -> pd.DataFrame:

        # Encode and scale samples
        samples_encoded, _ = self.dataPreparer.scale_and_encode(samples_df)

        if samples_encoded.shape[0] == 0:
            return pd.DataFrame(columns=self.columns)

        # Set number of threads for parallel processing
        if n_threads is None:
            n_threads = os.cpu_count() or 1

        def process_sample(sample_encoded):
            return self.find_closest_hull_point(
                sample_encoded,
                fixed_idx=fixed_idx,
                move_inside=move_inside
            )

        print("Generating samples")

        # Process samples in parallel
        with ThreadPoolExecutor(max_workers=n_threads) as executor:
            generated_samples = list(
                executor.map(process_sample, samples_encoded)
            )

        generated_samples_df = self.dataPreparer.unscale_and_decode(
            np.array(generated_samples)
        )

        return generated_samples_df

    def get_ray_directions(self, n_dimensions: int, n_rays: int) -> np.ndarray:
        dirs = []

        # Base directions
        for i in range(n_dimensions):
            e = np.zeros(n_dimensions)
            e[i] = 1
            dirs.append(e)
            dirs.append(-e)

        if len(dirs) >= n_rays:
            return np.array(dirs[:n_rays])

        # Fill remaining rays
        remaining = n_rays - len(dirs)
        for k in range(remaining):
            v = np.array([((k+1) * (j+3)) %
                         19 for j in range(n_dimensions)], float)

            norm = np.linalg.norm(v)

            if norm < 1e-12:
                # Fallback
                v = np.array([(k + 1) * (j + 2)
                             for j in range(n_dimensions)], float)
                norm = np.linalg.norm(v)

            v /= norm
            dirs.append(v)

        return np.array(dirs)

    def move_until_barrier(self, test_inside: callable, start: np.ndarray, direction: np.ndarray, step_size: float) -> np.ndarray:
        x = start.copy()

        last_inside = x.copy()
        while True:
            # Move in the direction and check boundaries [0,1] due to the min-max scaler
            candidate = x + step_size * direction
            candidate_clamped = np.clip(candidate, 0.0, 1.0)

            # Boundary was hit
            if not np.allclose(candidate, candidate_clamped):
                return last_inside

            # Check if candidate is still inside the polytope
            if not test_inside(candidate_clamped):
                return last_inside

            last_inside = candidate_clamped
            x = candidate_clamped

    def map_polytope(
            self,
            test_inside: callable,
            start_point: np.ndarray,
            n_rays: int = 50,
            step_size: float = 0.05) -> np.ndarray:

        n_dimensions = len(start_point)
        directions = self.get_ray_directions(n_dimensions, n_rays)

        boundary_points = []

        for d in directions:
            boundary = self.move_until_barrier(
                test_inside, start_point, d, step_size)
            boundary_points.append(boundary.copy())

        return np.array(boundary_points)

    def compute_convex_hull(
        self,
        points: np.ndarray,
        start_point: np.ndarray,
        tolerance: float = 1e-12,
    ) -> dict:
        if not isinstance(points, np.ndarray):
            points = np.asarray(points, dtype=float)
        if points.ndim != 2:
            raise ValueError("points must be shape (N, D)")

        # Drop rows with NaN or Inf
        finite_mask = np.isfinite(points).all(axis=1)
        pts = points[finite_mask]
        if pts.shape[0] == 0:
            raise RuntimeError("All points are non-finite")

        # Remove duplicates
        pts = np.unique(pts, axis=0)
        N, D = pts.shape

        # Center and check affine rank
        centroid = pts.mean(axis=0)
        A = pts - centroid
        rank = np.linalg.matrix_rank(A, tol=tolerance)

        if rank == 0:
            raise RuntimeError(
                "Points are all identical (rank 0). No hull to compute.")

        # Try full dimensional case
        if rank == D:
            hull = ConvexHull(pts)
            verts = hull.points[hull.vertices]
            return {
                "hull": hull,
                "centroid": centroid,
                "start_point": start_point,
                "V": None,
                "rank": D,
                "verts": verts,
            }

        # Project to lower dimension
        U, s, Vt = np.linalg.svd(A, full_matrices=False)
        r = rank
        V = Vt.T[:, :r]
        projected = A @ V

        hull_proj = ConvexHull(projected)
        verts = hull_proj.points[hull_proj.vertices]

        return {
            "hull": hull_proj,
            "centroid": centroid,
            "start_point": start_point,
            "V": V,
            "rank": r,
            "verts": verts,
        }

    def is_inside_hull(
        self,
        point: np.ndarray,
        hull_info: dict,
        tol: float = 1e-12,
    ) -> bool:
        hull = hull_info["hull"]

        point = np.asarray(point, dtype=float)

        # Project point if hull is low-rank
        if hull_info["V"] is not None:
            point = (point - hull_info["centroid"]) @ hull_info["V"]

        return np.all(
            hull.equations[:, :-1] @ point + hull.equations[:, -1] <= tol
        )

    def closest_point_in_hull(self, hull_info: dict, point: np.ndarray, fixed_idx: tuple = (), move_inside: float = 0.0) -> np.ndarray:
        V = hull_info["V"]
        centroid = hull_info["centroid"]
        start_point = hull_info["start_point"]
        verts = hull_info["verts"]

        m, _ = verts.shape
        w = cp.Variable(m)

        # Project point if hull is low-rank
        if V is not None:
            p_proj = (point - centroid) @ V
        else:
            p_proj = point

        # Add convex combination constraints
        constraints = [w >= 0, cp.sum(w) == 1]

        # Add fixed index constraints if specified (functional features)
        if fixed_idx:
            fixed_idx = np.asarray(fixed_idx, dtype=int)
            constraints.append(
                w @ verts[:, fixed_idx] == p_proj[fixed_idx]
            )

        # Create objective
        reconstructed = w @ verts
        objective = cp.Minimize(cp.sum_squares(reconstructed - p_proj))
        prob = cp.Problem(objective, constraints)

        prob.solve(solver=cp.OSQP)

        if w.value is None:
            raise RuntimeError("Solver failed")

        closest_proj = w.value @ verts

        # Reconstruct full point if hull was low-rank
        if V is not None:
            closest_point = closest_proj @ V.T + centroid
        else:
            closest_point = closest_proj

        # Move slightly inside the hull if requested, to ensure we are not on the boundary
        if move_inside > 0.0:
            direction_to_cluster_center = start_point - closest_point
            closest_point += move_inside * direction_to_cluster_center

        return closest_point
