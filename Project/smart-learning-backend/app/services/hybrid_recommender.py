import logging
import pickle
import re
import unicodedata
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch
from pyvi import ViTokenizer
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

logger = logging.getLogger(__name__)

# Thư mục gốc backend (chứa các file .pkl)
_BACKEND_DIR = Path(__file__).resolve().parent.parent.parent


class HybridTeacherRecommender:
    """
    Hệ thống gợi ý Hybrid kết hợp:
    1. Content-Based (Sentence Transformers + PyVi)
    2. Collaborative Filtering (SVD)
    """

    def __init__(self, backend_dir: Path | None = None):
        base = backend_dir or _BACKEND_DIR

        # 1. Nạp mô hình SVD (Collaborative Filtering)
        svd_path = base / "teacher_svd_model.pkl"
        self.svd_model = joblib.load(svd_path)
        logger.info("Loaded SVD model from %s", svd_path)

        # 2. Nạp ma trận Vector Giảng viên (Content-Based)
        emb_path = base / "teacher_embeddings.pkl"
        with open(emb_path, "rb") as f:
            self.teacher_embeddings = pickle.load(f)  # noqa: S301 – trusted local file

        self.teacher_ids = list(self.teacher_embeddings.keys())
        self.teacher_vectors = np.array(list(self.teacher_embeddings.values()))
        logger.info("Loaded %d teacher embedding vectors.", len(self.teacher_ids))

        # 3. Nạp thông tin môn học từ CSV
        csv_path = base / ".." / "Data" / "teacher_profiles_cleaned.csv"
        csv_path = csv_path.resolve()
        if csv_path.exists():
            df = pd.read_csv(csv_path)
            self.teacher_names = df.set_index('teacher_id')['teacher_name'].to_dict()
            if set(self.teacher_ids) != set(df.teacher_id):
                raise ValueError('Embedding IDs do not match the teacher catalog')
            links = pd.read_csv(csv_path.parent / 'teacher_courses.csv')
            self.course_teachers = {}
            for row in links.itertuples():
                key = self._normalize_text(row.course_name)
                self.course_teachers.setdefault(key, set()).add(int(row.teacher_id))
            self.teacher_courses: dict[int, str] = (
                df.set_index("teacher_id")["courses_taught"]
                .fillna("")
                .to_dict()
            )
            logger.info("Loaded courses for %d teachers from CSV.", len(self.teacher_courses))
        else:
            self.teacher_names = {}
            self.teacher_courses = {}
            logger.warning("CSV not found at %s – course_name will be empty.", csv_path)

        # 4. Khởi tạo mô hình NLP Tiếng Việt
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.nlp_model = SentenceTransformer(
            "keepitreal/vietnamese-sbert", device=self.device, local_files_only=True
        )
        logger.info("NLP model ready on %s.", self.device.upper())

    @staticmethod
    def _normalize_text(text: str) -> str:
        text = unicodedata.normalize('NFD', text.casefold().replace('đ', 'd'))
        text = ''.join(c for c in text if unicodedata.category(c) != 'Mn')
        # Optional preposition: "lập trình [cho] thiết bị di động".
        return ' '.join(word for word in re.findall(r'\w+', text) if word != 'cho')

    @staticmethod
    def _normalize_svd(est: float, lo: float = 1.0, hi: float = 10.0) -> float:
        """Chuẩn hóa điểm SVD từ thang 1-10 về thang 0-1, clamp về [0, 1]."""
        return max(0.0, min(1.0, (est - lo) / (hi - lo)))

    def recommend(
        self,
        student_id: str | int | None,
        query_text: str,
        alpha: float = 0.5,
        top_k: int = 5,
    ) -> list[dict]:
        """
        Thực thi thuật toán Hybrid.
        - alpha: Trọng số Content-Based (0.0 → chỉ SVD, 1.0 → chỉ NLP).
        """
        # ── Content-Based ──
        processed_query = ViTokenizer.tokenize(query_text)
        query_vector = self.nlp_model.encode([processed_query])
        cosine_scores = cosine_similarity(query_vector, self.teacher_vectors)[0]
        cb_map = dict(zip(self.teacher_ids, cosine_scores))

        # Use recorded course names as an explicit constraint when mentioned.
        # Prefer the longer title over a title nested within it.
        query = ' ' + self._normalize_text(query_text) + ' '
        matches = [name for name in self.course_teachers if ' ' + name + ' ' in query]
        matches = [name for name in matches if not any(name != other and ' '+name+' ' in ' '+other+' ' for other in matches)]
        eligible = set().union(*(self.course_teachers[name] for name in matches)) if matches else set(self.teacher_ids)

        # ── Hybrid ──
        results = []
        for t_id in self.teacher_ids:
            if t_id not in eligible:
                continue
            cf_est = self.svd_model.predict(uid=str(student_id) if student_id is not None else None, iid=t_id).est
            cf_score = self._normalize_svd(cf_est)
            cb_score = cb_map[t_id]
            hybrid = alpha * cb_score + (1 - alpha) * cf_score

            results.append({
                "teacher_id": t_id,
                "teacher_name": self.teacher_names.get(t_id, ""),
                "content_score": round(float(cb_score), 4),
                "collab_score": round(float(cf_score), 4),
                "hybrid_score": round(float(hybrid), 4),
                "course_name": self.teacher_courses.get(t_id, ""),
            })

        results.sort(key=lambda x: x["hybrid_score"], reverse=True)
        return results[:top_k]
