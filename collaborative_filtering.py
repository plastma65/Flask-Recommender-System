"""
Collaborative Filtering Module (STU‑CF) – Stand‑alone (Fixed)
- Adds missing evaluate() method
- Fixes item_norm broadcasting bug and typo
- Hardens _truncate_top_k against k > n
- Minor clarity tweaks in messages and denominators
"""
from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from sklearn.decomposition import TruncatedSVD
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    mean_squared_error,
    recall_score,
    roc_auc_score,
)
from sklearn.metrics.pairwise import cosine_similarity

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(levelname)s – %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)


class CollaborativeFilteringCF:
    """Memory-efficient User-User / Item-Item CF + SVD MF."""

    def __init__(
        self,
        k_neighbors: int = 15,
        threshold_pass: float = 5.0,
        svd_components: int = 100,
        svd_random_state: int = 42,
    ) -> None:
        # KNN CF parameters
        self.k_neighbors = k_neighbors
        self.threshold_pass = threshold_pass
        # Data placeholders
        self.user_item_df: pd.DataFrame | None = None
        self.user_item_matrix: csr_matrix | None = None
        self.user_means: np.ndarray | None = None
        self.item_means: np.ndarray | None = None
        # Similarity matrices (for CF)
        self.user_sim: np.ndarray | None = None
        self.item_sim: np.ndarray | None = None
        self.course_stats: pd.DataFrame | None = None
        # SVD parameters and factors
        self.svd_components = svd_components
        self.svd_random_state = svd_random_state
        self.svd_model: TruncatedSVD | None = None
        self.user_factors: np.ndarray | None = None
        self.item_factors: np.ndarray | None = None
        # Prerequisite map for courses
        self.prereq_map: Dict[str, List[str]] = {}

    # ------------------------------------------------------------------ #
    #  Data loading & preprocessing
    # ------------------------------------------------------------------ #
    def load_data(self, scores_path: str, survey_path: str | None = None) -> pd.DataFrame:
        df = pd.read_csv(scores_path, dtype={"student_id": str, "course_code": str})
        df = df.dropna(subset=["student_id", "course_code", "total_score"])
        df = df.loc[df["total_score"].between(0, 10)]

        if survey_path:
            survey = pd.read_csv(survey_path)
            course_cols = {"course_code", "difficulty", "pass_rate", "workload"}
            if course_cols <= set(survey.columns):
                df = df.merge(
                    survey[list(course_cols)],
                    on="course_code",
                    how="left",
                    suffixes=("", "_course"),
                )

        self.course_stats = (
            df.groupby("course_code")["total_score"].agg(avg_score="mean", student_count="count").reset_index()
        )
        return df

    def create_user_item_matrix(self, df: pd.DataFrame) -> None:
        pivot = (
            df.pivot_table(index="student_id", columns="course_code", values="total_score", aggfunc="mean")
            .fillna(0.0)
            .astype(np.float32)
        )
        self.user_item_df = pivot
        self.user_item_matrix = csr_matrix(pivot.values)
        self.user_means = pivot.mean(axis=1).to_numpy(dtype=np.float32)
        self.item_means = pivot.mean(axis=0).to_numpy(dtype=np.float32)

        sparsity = 1.0 - (pivot.astype(bool).sum().sum() / pivot.size)
        logging.info("User-Item matrix %s – sparsity %.2f%%", pivot.shape, sparsity * 100)

    # ------------------------------------------------------------------ #
    #  Similarities (for CF)
    # ------------------------------------------------------------------ #
    @staticmethod
    def _truncate_top_k(sim: np.ndarray, k: int) -> np.ndarray:
        """Zero out all but the top-k highest similarity values in each row.
        Handles k > n_cols gracefully.
        """
        if sim.ndim != 2:
            return sim
        n_rows, n_cols = sim.shape
        k = max(1, min(k, n_cols))
        idx = np.argpartition(sim, -k, axis=1)[:, -k:]
        mask = np.zeros_like(sim, dtype=bool)
        rows = np.arange(n_rows)[:, None]
        mask[rows, idx] = True
        return np.where(mask, sim, 0.0)

    def compute_similarities(self) -> None:
        if self.user_item_matrix is None:
            raise RuntimeError("Call create_user_item_matrix first.")

        ratings = self.user_item_matrix.toarray()
        # User-User similarity: normalize by user means
        user_norm = ratings - self.user_means[:, None]
        user_norm[ratings == 0] = 0
        sim_u = cosine_similarity(user_norm)
        np.fill_diagonal(sim_u, 0.0)
        self.user_sim = self._truncate_top_k(sim_u, self.k_neighbors)

        # Item-Item similarity: normalize by item means (broadcast over columns)
        item_norm = ratings - self.item_means[None, :]
        item_norm[ratings == 0] = 0
        sim_i = cosine_similarity(item_norm.T)
        np.fill_diagonal(sim_i, 0.0)
        self.item_sim = self._truncate_top_k(sim_i, self.k_neighbors)

    # ------------------------------------------------------------------ #
    #  Predictions (CF)
    # ------------------------------------------------------------------ #
    def predict_user_based(self) -> np.ndarray:
        if self.user_sim is None:
            raise RuntimeError("Call compute_similarities() first.")
        if self.user_item_matrix is None:
            raise RuntimeError("User-item matrix is not initialized. Call create_user_item_matrix() first.")
        ratings = self.user_item_matrix.toarray()
        centered = ratings - self.user_means[:, None]
        centered[ratings == 0] = 0
        numer = self.user_sim @ centered
        denom = np.maximum(self.user_sim.sum(axis=1, keepdims=True), 1e-12)
        pred = self.user_means[:, None] + numer / denom
        pred[ratings > 0] = ratings[ratings > 0]
        return np.clip(pred, 0, 10)

    def predict_item_based(self) -> np.ndarray:
        if self.item_sim is None:
            raise RuntimeError("Call compute_similarities() first.")
        if self.user_item_matrix is None:
            raise RuntimeError("User-item matrix is not initialized. Call create_user_item_matrix() first.")
        ratings = self.user_item_matrix.toarray()
        numer = ratings @ self.item_sim
        # sum of absolute similarities avoids sign-cancel and zeros
        denom = np.maximum(np.abs(self.item_sim).sum(axis=1), 1e-12)
        pred = numer / denom
        pred[ratings > 0] = ratings[ratings > 0]
        return np.clip(pred, 0, 10)

    @staticmethod
    def combine_predictions(user_pred: np.ndarray, item_pred: np.ndarray, alpha: float = 0.6) -> np.ndarray:
        return alpha * user_pred + (1 - alpha) * item_pred

    # ------------------------------------------------------------------ #
    #  Matrix Factorization via Truncated SVD (latent factors)
    # ------------------------------------------------------------------ #
    def compute_svd(self) -> None:
        if self.user_item_matrix is None or self.user_item_df is None or self.user_means is None:
            raise RuntimeError("Call create_user_item_matrix first.")
        ratings = self.user_item_matrix.toarray().astype(np.float32)
        mask = ratings > 0
        centered = ratings - self.user_means[:, None]
        centered[~mask] = 0.0
        n_users, n_items = centered.shape
        k_max = max(1, min(n_users, n_items) - 1)
        k = min(int(self.svd_components), k_max)
        if k < 2:
            logging.warning("Not enough data for SVD (users=%d, items=%d) → skip.", n_users, n_items)
            self.svd_model = None
            self.user_factors = None
            self.item_factors = None
            return
        svd = TruncatedSVD(n_components=k, random_state=self.svd_random_state)
        self.user_factors = svd.fit_transform(centered).astype(np.float32)
        self.item_factors = svd.components_.T.astype(np.float32)
        self.svd_model = svd
        logging.info(
            "TruncatedSVD trained with k=%d (max=%d) – explained_variance=%.2f%%",
            k,
            k_max,
            100.0 * svd.explained_variance_ratio_.sum(),
        )

    def predict_svd(self) -> np.ndarray:
        if self.user_item_matrix is None or self.user_factors is None or self.item_factors is None:
            raise RuntimeError("Call compute_svd() first.")
        ratings = self.user_item_matrix.toarray()
        reconstructed_centered = self.user_factors @ self.item_factors.T
        pred = reconstructed_centered + self.user_means[:, None]
        pred[ratings > 0] = ratings[ratings > 0]
        return np.clip(pred, 0, 10)

    # ------------------------------------------------------------------ #
    #  Recommend single user
    # ------------------------------------------------------------------ #
    def recommend(self, student_id: str, predictions: np.ndarray, n: int = 5) -> pd.DataFrame | str:
        if self.user_item_df is None:
            raise RuntimeError("Matrix not built.")
        if student_id not in self.user_item_df.index:
            return f"Student {student_id} not found."
        uid = self.user_item_df.index.get_loc(student_id)
        rated = self.user_item_df.iloc[uid].to_numpy(dtype=float) > 0
        scores = predictions[uid]
        unrated = np.where(~rated)[0]
        if unrated.size == 0:
            return "Student has rated all courses."
        allowed_indices: List[int] = []
        for idx in unrated:
            code = self.user_item_df.columns[idx]
            prereqs = self.prereq_map.get(code, [])
            if prereqs:
                prereq_satisfied = True
                for pre in prereqs:
                    if pre not in self.user_item_df.columns:
                        prereq_satisfied = False
                        break
                    pre_score = float(self.user_item_df.at[student_id, pre])
                    if pre_score < self.threshold_pass:
                        prereq_satisfied = False
                        break
                if not prereq_satisfied:
                    continue
            allowed_indices.append(idx)
        if not allowed_indices:
            return "No eligible courses after applying prerequisites."
        unrated = np.array(allowed_indices, dtype=int)
        top = unrated[np.argpartition(scores[unrated], -n)[-n:]]
        top = top[np.argsort(scores[top])[::-1]]
        recs = []
        for rank, idx in enumerate(top, 1):
            code = self.user_item_df.columns[idx]
            recs.append(
                {
                    "rank": rank,
                    "course_code": code,
                    "predicted_score": round(float(scores[idx]), 2),
                    "pass_probability": round(float(1 / (1 + np.exp(-(scores[idx] - self.threshold_pass)))), 3),
                }
            )
        return pd.DataFrame(recs)

    # ------------------------------------------------------------------ #
    #  Batch recommend
    # ------------------------------------------------------------------ #
    def batch_recommend(self, predictions: np.ndarray, top_n: int = 5) -> pd.DataFrame:
        if self.user_item_df is None:
            raise RuntimeError("Matrix not built.")
        rows = []
        for sid in self.user_item_df.index:
            rec = self.recommend(sid, predictions, n=top_n)
            if isinstance(rec, pd.DataFrame):
                rec["student_id"] = sid
                rows.append(rec)
        if not rows:
            return pd.DataFrame(columns=["student_id", "rank", "course_code", "predicted_score", "pass_probability"])
        df_out = pd.concat(rows, ignore_index=True)
        cols = ["student_id", "rank", "course_code", "predicted_score", "pass_probability"]
        return df_out[cols]

    # ------------------------------------------------------------------ #
    #  Evaluation (added)
    # ------------------------------------------------------------------ #
    def evaluate(self, actual: np.ndarray, pred: np.ndarray) -> Dict[str, Dict[str, float]]:
        """Compute RMSE and pass/fail classification metrics using threshold_pass."""
        if actual.shape != pred.shape:
            raise ValueError(f"Shape mismatch: actual {actual.shape} vs pred {pred.shape}")
        rmse = float(np.sqrt(mean_squared_error(actual.flatten(), pred.flatten())))
        # Binarize using threshold
        y_true = (actual >= self.threshold_pass).astype(int).ravel()
        y_score = pred.ravel()
        y_pred = (y_score >= self.threshold_pass).astype(int)
        acc = float(accuracy_score(y_true, y_pred))
        rec = float(recall_score(y_true, y_pred, zero_division=0))
        f1 = float(f1_score(y_true, y_pred, zero_division=0))
        # ROC AUC requires both classes; guard to avoid ValueError
        try:
            auc = float(roc_auc_score(y_true, y_score))
        except ValueError:
            auc = float("nan")
        return {
            "regression": {"rmse": rmse},
            "classification": {"accuracy": acc, "recall": rec, "f1": f1, "roc_auc": auc},
        }

    # ------------------------------------------------------------------ #
    #  Cross-validation utility (optional for research)
    # ------------------------------------------------------------------ #
    def cross_validate(self, df: pd.DataFrame, n_splits: int = 5) -> List[Dict[str, Dict[str, float]]]:
        students = df["student_id"].unique()
        np.random.shuffle(students)
        fold_sizes = np.full(n_splits, len(students) // n_splits, dtype=int)
        fold_sizes[: len(students) % n_splits] += 1
        results = []
        idx = 0
        for fold, size in enumerate(fold_sizes, 1):
            test_ids = students[idx : idx + size]
            idx += size
            # Train on (all data minus test_ids)
            train_df = df[~df["student_id"].isin(test_ids)]
            self.create_user_item_matrix(train_df)
            self.compute_similarities()
            user_pred = self.predict_user_based()
            item_pred = self.predict_item_based()
            comb_pred = self.combine_predictions(user_pred, item_pred)
            self.compute_svd()
            if self.user_factors is not None and self.item_factors is not None:
                svd_pred = self.predict_svd()
            else:
                svd_pred = comb_pred
                logging.info("Fold %d: not enough data for SVD, using CF predictions only.", fold)
            blend_pred = 0.5 * comb_pred + 0.5 * svd_pred
            # Re-create full user-item matrix to evaluate on test fold
            self.create_user_item_matrix(df)
            if self.user_item_matrix is None or self.user_item_df is None:
                raise RuntimeError("User-item matrix or DataFrame is not initialized. Call create_user_item_matrix() first.")
            test_mask = self.user_item_df.index.isin(test_ids)
            actual = self.user_item_matrix.toarray()[test_mask]
            pred = blend_pred[test_mask]
            results.append(self.evaluate(actual, pred))
            logging.info("Fold %d/%d – RMSE %.4f", fold, n_splits, results[-1]["regression"]["rmse"])
        return results


# ---------------------------------------------------------------------- #
#  CLI
# ---------------------------------------------------------------------- #

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Collaborative Filtering – STU module (+SVD)")
    p.add_argument("scores_csv", help="CSV with student-course total_score")
    p.add_argument("--survey_csv", help="Optional survey CSV", default=None)
    p.add_argument(
        "--mode",
        choices=["user", "item", "combined", "svd", "blend"],
        default="combined",
        help="Prediction mode: KNN user/item/combined, SVD, or blend of combined+svd",
    )
    p.add_argument("--top", type=int, default=5, help="Top-N recommendations per student")
    p.add_argument("--output", default="cf_recommendations.csv", help="CSV path for output recommendations")
    p.add_argument("--metrics_json", help="Optional path to save evaluation metrics as JSON")
    p.add_argument("--svd_components", type=int, default=100, help="Number of SVD latent dimensions (default: 100)")
    p.add_argument("--svd_alpha", type=float, default=0.5, help="Blend weight for SVD in --mode blend (0-1)")
    p.add_argument("--prereq_csv", help="CSV file of course prerequisites (optional)", default=None)
    p.add_argument("--userid", help="Only output recommendations for this student (optional)", default=None)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    cf = CollaborativeFilteringCF(svd_components=args.svd_components)
    df_raw = cf.load_data(args.scores_csv, args.survey_csv)
    cf.create_user_item_matrix(df_raw)
    cf.compute_similarities()
    # Load prerequisite map from CSV if provided
    if args.prereq_csv:
        preq_df = pd.read_csv(args.prereq_csv, dtype=str).fillna("")
        prereq_map: Dict[str, List[str]] = {}
        for _, row in preq_df.iterrows():
            course = str(row["course_code"]).strip().upper()
            prereq_field = str(row["prerequisite"]).strip() if not pd.isna(row["prerequisite"]) else ""
            if prereq_field:
                prereqs = [x.strip().upper() for x in prereq_field.split(",") if x.strip()]
            else:
                prereqs = []
            if prereqs:
                prereq_map[course] = prereqs
        cf.prereq_map = prereq_map

    # Compute predictions according to selected mode
    user_pred = cf.predict_user_based()
    item_pred = cf.predict_item_based()
    comb_pred = cf.combine_predictions(user_pred, item_pred)
    svd_pred = None
    if args.mode in {"svd", "blend"}:
        cf.compute_svd()
        if cf.user_factors is not None and cf.item_factors is not None:
            svd_pred = cf.predict_svd()
        else:
            logging.warning("Skipping SVD predictions due to insufficient data.")
            svd_pred = None

    if args.mode == "user":
        predictions = user_pred
    elif args.mode == "item":
        predictions = item_pred
    elif args.mode == "combined":
        predictions = comb_pred
    elif args.mode == "svd":
        predictions = svd_pred if svd_pred is not None else comb_pred
    else:  # blend
        predictions = (1 - args.svd_alpha) * comb_pred + args.svd_alpha * svd_pred if svd_pred is not None else comb_pred

    rec_df = cf.batch_recommend(predictions, top_n=args.top)
    if args.userid:
        user_id_str = str(args.userid)
        if user_id_str not in cf.user_item_df.index:
            logging.error("Student %s not found in data.", user_id_str)
        rec_df = rec_df[rec_df["student_id"] == user_id_str]

    rec_df.to_csv(args.output, index=False, encoding="utf-8-sig")
    logging.info("Saved %d recommendations → %s", len(rec_df), Path(args.output).resolve())

    metrics = cf.evaluate(cf.user_item_matrix.toarray(), predictions)
    logging.info(
        "Global RMSE %.4f – Accuracy %.3f", metrics["regression"]["rmse"], metrics["classification"]["accuracy"]
    )
    if args.metrics_json:
        Path(args.metrics_json).write_text(json.dumps(metrics, indent=2, ensure_ascii=False))
        logging.info("Metrics saved → %s", Path(args.metrics_json).resolve())


if __name__ == "__main__":
    main()
