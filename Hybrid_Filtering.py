
"""
Collaborative Filtering Module (STU‑CF) – Stand‑alone
=====================================================
BẢN NÂNG CẤP: Bổ sung Matrix Factorization bằng Truncated SVD (k=100)
- Thêm chế độ --mode svd để dự đoán bằng SVD.
- Giữ nguyên các chế độ user, item, combined.
- Có thể pha trộn SVD với combined bằng --svd_alpha (tuỳ chọn).
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Dict, List, Optional

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
    """Memory‑efficient User‑User / Item‑Item CF + SVD MF (k=100)."""

    def __init__(
        self,
        k_neighbors: int = 15,
        threshold_pass: float = 5.0,
        svd_components: int = 100,
        svd_random_state: int = 42,
    ) -> None:
        # KNN CF
        self.k_neighbors = k_neighbors
        self.threshold_pass = threshold_pass

        # Dữ liệu
        self.user_item_df: pd.DataFrame | None = None
        self.user_item_matrix: csr_matrix | None = None
        self.user_means: np.ndarray | None = None
        self.item_means: np.ndarray | None = None

        # Similarities
        self.user_sim: np.ndarray | None = None
        self.item_sim: np.ndarray | None = None
        self.course_stats: pd.DataFrame | None = None

        # SVD (MF)
        self.svd_components = svd_components
        self.svd_random_state = svd_random_state
        self.svd_model: Optional[TruncatedSVD] = None
        self.user_factors: Optional[np.ndarray] = None  # shape (n_users, k)
        self.item_factors: Optional[np.ndarray] = None  # shape (n_items, k)

    # ------------------------------------------------------------------ #
    #  Data loading & preprocessing
    # ------------------------------------------------------------------ #
    def load_data(self, scores_path: str, survey_path: str | None = None) -> pd.DataFrame:
        """Load & merge datasets (survey optional)."""
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
            df.groupby("course_code")["total_score"]
            .agg(avg_score="mean", student_count="count")
            .reset_index()
        )
        return df

    def create_user_item_matrix(self, df: pd.DataFrame) -> None:
        pivot = (
            df.pivot_table(
                index="student_id",
                columns="course_code",
                values="total_score",
                aggfunc="mean",
            )
            .fillna(0.0)
            .astype(np.float32)
        )
        self.user_item_df = pivot
        self.user_item_matrix = csr_matrix(pivot.values)
        self.user_means = pivot.mean(axis=1).to_numpy(dtype=np.float32)
        self.item_means = pivot.mean(axis=0).to_numpy(dtype=np.float32)

        sparsity = 1.0 - (pivot.astype(bool).sum().sum() / pivot.size)
        logging.info("User‑Item matrix %s – sparsity %.2f%%", pivot.shape, sparsity * 100)

    # ------------------------------------------------------------------ #
    #  Similarities (KNN CF)
    # ------------------------------------------------------------------ #
    @staticmethod
    def _truncate_top_k(sim: np.ndarray, k: int) -> np.ndarray:
        idx = np.argpartition(sim, -k, axis=1)[:, -k:]
        mask = np.zeros_like(sim, dtype=bool)
        rows = np.arange(sim.shape[0])[:, None]
        mask[rows, idx] = True
        return np.where(mask, sim, 0.0)

    def compute_similarities(self) -> None:
        if self.user_item_matrix is None:
            raise RuntimeError("Call create_user_item_matrix first.")

        ratings = self.user_item_matrix.toarray()
        # User‑User: mean‑center theo người dùng
        user_norm = ratings - self.user_means[:, None]
        user_norm[ratings == 0] = 0
        sim_u = cosine_similarity(user_norm)
        np.fill_diagonal(sim_u, 0.0)
        self.user_sim = self._truncate_top_k(sim_u, self.k_neighbors)

        # Item‑Item: mean‑center theo môn học
        item_norm = (ratings - self.item_means)[..., None].squeeze()
        item_norm[ratings == 0] = 0
        sim_i = cosine_similarity(item_norm.T)
        np.fill_diagonal(sim_i, 0.0)
        self.item_sim = self._truncate_top_k(sim_i, self.k_neighbors)

    # ------------------------------------------------------------------ #
    #  Predictions (KNN CF)
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
        denom = np.maximum(self.item_sim.sum(axis=1), 1e-12)
        pred = numer / denom
        pred[ratings > 0] = ratings[ratings > 0]
        return np.clip(pred, 0, 10)

    @staticmethod
    def combine_predictions(user_pred: np.ndarray, item_pred: np.ndarray, alpha: float = 0.6) -> np.ndarray:
        return alpha * user_pred + (1 - alpha) * item_pred

    # ------------------------------------------------------------------ #
    #  Matrix Factorization via Truncated SVD (k=100)
    # ------------------------------------------------------------------ #
    def compute_svd(self) -> None:
        """Huấn luyện TruncatedSVD trên ma trận đã mean‑center theo user.
        Lưu user_factors (U) và item_factors (V)."""
        if self.user_item_matrix is None or self.user_item_df is None or self.user_means is None:
            raise RuntimeError("Call create_user_item_matrix first.")

        ratings = self.user_item_matrix.toarray().astype(np.float32)  # (n_users, n_items)
        # Mean‑center theo user chỉ trên các ô đã có rating
        centered = ratings.copy()
        centered[ratings > 0] -= self.user_means[:, None]

        k = min(self.svd_components, min(centered.shape) - 1)
        if k < 2:
            raise RuntimeError("Not enough data to compute SVD.")

        svd = TruncatedSVD(n_components=k, random_state=self.svd_random_state)
        user_latent = svd.fit_transform(centered)  # shape (n_users, k)
        item_latent = svd.components_.T            # shape (n_items, k)

        self.svd_model = svd
        self.user_factors = user_latent.astype(np.float32)
        self.item_factors = item_latent.astype(np.float32)
        logging.info("TruncatedSVD trained: k=%d, explained_variance=%.2f%%",
                     k, 100.0 * svd.explained_variance_ratio_.sum())

    def predict_svd(self) -> np.ndarray:
        if self.user_item_matrix is None or self.user_factors is None or self.item_factors is None:
            raise RuntimeError("Call compute_svd() first.")
        ratings = self.user_item_matrix.toarray()
        # Khôi phục ma trận dự đoán từ nhân tử hoá và cộng lại user_means
        reconstructed_centered = self.user_factors @ self.item_factors.T  # (n_users, n_items)
        pred = reconstructed_centered + self.user_means[:, None]
        # Giữ nguyên các ô đã biết
        pred[ratings > 0] = ratings[ratings > 0]
        return np.clip(pred, 0, 10)

    # ------------------------------------------------------------------ #
    #  Evaluation
    # ------------------------------------------------------------------ #
    def evaluate(self, actual: np.ndarray, pred: np.ndarray) -> Dict[str, Dict[str, float]]:
        mask = actual > 0
        y_true = actual[mask]
        y_pred = pred[mask]

        mse = mean_squared_error(y_true, y_pred)
        metrics = {
            "regression": {
                "rmse": float(np.sqrt(mse)),
                "mae": float(np.abs(y_true - y_pred).mean()),
            },
            "classification": {},
        }
        y_true_bin = (y_true >= self.threshold_pass).astype(int)
        y_pred_bin = (y_pred >= self.threshold_pass).astype(int)
        metrics["classification"].update(
            accuracy=float(accuracy_score(y_true_bin, y_pred_bin)),
            f1=float(f1_score(y_true_bin, y_pred_bin, zero_division=0)),
            recall=float(recall_score(y_true_bin, y_pred_bin, zero_division=0)),
        )
        if len(np.unique(y_true_bin)) > 1:
            proba = 1 / (1 + np.exp(-(y_pred - self.threshold_pass)))
            metrics["classification"]["roc_auc"] = float(roc_auc_score(y_true_bin, proba))
        return metrics

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
                    "pass_probability": round(
                        float(1 / (1 + np.exp(-(scores[idx] - self.threshold_pass)))), 3
                    ),
                }
            )
        return pd.DataFrame(recs)

    # ------------------------------------------------------------------ #
    #  Batch recommend
    # ------------------------------------------------------------------ #
    def batch_recommend(self, predictions: np.ndarray, top_n: int = 5) -> pd.DataFrame:
        rows = []
        for sid in self.user_item_df.index:  # type: ignore[union-attr]
            rec = self.recommend(sid, predictions, n=top_n)
            if isinstance(rec, pd.DataFrame):
                rec["student_id"] = sid
                rows.append(rec)
        df_out = pd.concat(rows, ignore_index=True)
        cols = ["student_id", "rank", "course_code", "predicted_score", "pass_probability"]
        return df_out[cols]

    # ------------------------------------------------------------------ #
    #  Cross‑validation utility (optional for research)
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

            train_df = df[~df["student_id"].isin(test_ids)]
            self.create_user_item_matrix(train_df)
            self.compute_similarities()
            user_pred = self.predict_user_based()
            item_pred = self.predict_item_based()
            comb_pred = self.combine_predictions(user_pred, item_pred)

            # SVD trên toàn ma trận huấn luyện
            self.compute_svd()
            svd_pred = self.predict_svd()

            # Pha trộn đơn giản giữa combined và svd (50/50) cho đánh giá nghiên cứu
            blend_pred = 0.5 * comb_pred + 0.5 * svd_pred

            # Re-create user_item_df and user_item_matrix for full data để mask test
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
    p.add_argument("scores_csv", help="CSV with student‑course total_score")
    p.add_argument("--survey_csv", help="Optional survey CSV", default=None)
    p.add_argument(
        "--mode",
        choices=["user", "item", "combined", "svd", "blend"],
        default="combined",
        help="Prediction mode: KNN user/item/combined, SVD MF, or blend of combined+svd",
    )
    p.add_argument("--top", type=int, default=5, help="Top‑N recommendations per student")
    p.add_argument("--output", default="cf_recommendations.csv", help="CSV path for Moodle upload")
    p.add_argument("--metrics_json", help="Optional path to save evaluation metrics as JSON")
    p.add_argument("--svd_components", type=int, default=100, help="Number of SVD latent dims (default 100)")
    p.add_argument("--svd_alpha", type=float, default=0.5, help="Blend weight for SVD in --mode blend (0..1)")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    cf = CollaborativeFilteringCF(svd_components=args.svd_components)
    df_raw = cf.load_data(args.scores_csv, args.survey_csv)
    cf.create_user_item_matrix(df_raw)
    cf.compute_similarities()

    # KNN CF predictions
    user_pred = cf.predict_user_based()
    item_pred = cf.predict_item_based()
    comb_pred = cf.combine_predictions(user_pred, item_pred)

    # SVD predictions (train lazily when needed)
    if args.mode in {"svd", "blend"}:
        cf.compute_svd()
        svd_pred = cf.predict_svd()

    if args.mode == "user":
        predictions = user_pred
    elif args.mode == "item":
        predictions = item_pred
    elif args.mode == "combined":
        predictions = comb_pred
    elif args.mode == "svd":
        predictions = svd_pred  # type: ignore[name-defined]
    else:  # blend
        # Pha trộn giữa combined (KNN) và SVD theo svd_alpha
        predictions = (1 - args.svd_alpha) * comb_pred + args.svd_alpha * svd_pred  # type: ignore[name-defined]

    # Export recommendations
    rec_df = cf.batch_recommend(predictions, top_n=args.top)
    rec_df.to_csv(args.output, index=False, encoding="utf-8-sig")
    logging.info("Saved %d recommendations → %s", len(rec_df), Path(args.output).resolve())

    # Evaluate on full dataset (optional)
    metrics = cf.evaluate(cf.user_item_matrix.toarray(), predictions)  # type: ignore[arg-type]
    logging.info(
        "Global RMSE %.4f – Accuracy %.3f",
        metrics["regression"]["rmse"],
        metrics["classification"]["accuracy"],
    )
    if args.metrics_json:
        Path(args.metrics_json).write_text(json.dumps(metrics, indent=2, ensure_ascii=False))
        logging.info("Metrics saved → %s", Path(args.metrics_json).resolve())


if __name__ == "__main__":
    main()
