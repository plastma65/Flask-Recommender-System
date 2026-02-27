from sentence_transformers import SentenceTransformer
import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from typing import List, Tuple, Optional
from underthesea import word_tokenize

# Các cột văn bản mặc định dùng để kết hợp thông tin khóa học
DEFAULT_TEXT_COLS = [
    "course_name",
    "course_category",
    "course_description",
    "semester",
]

def preprocess_text(text: str) -> str:
    """
    Tiền xử lý văn bản: kiểm tra NaN, tách từ bằng underthesea.
    Không dùng unidecode, không chuyển về chữ thường.
    """
    if pd.isna(text):
        return ""
    tokens = word_tokenize(str(text), format="text")
    return " ".join(tokens)

def build_combined_text(row: pd.Series, text_cols: List[str]) -> str:
    """
    Kết hợp các trường văn bản thành một chuỗi duy nhất, phân tách bằng dấu |
    """
    parts = [str(row.get(col, "")) for col in text_cols if col in row and pd.notna(row[col])]
    return " | ".join(parts)

class CourseRecommender:
    """
    Lớp gợi ý khóa học dựa trên nội dung (content-based filtering)
    """
    def __init__(self, text_cols: List[str] = None, model_name: str = "output_finetuned_vn"):
        # Các cột văn bản sử dụng, và tên model embedding
        self.text_cols = text_cols or DEFAULT_TEXT_COLS
        self.model = SentenceTransformer(model_name)
        self.course_embeddings: Optional[np.ndarray] = None
        self.course_ids: Optional[List[str]] = None
        self.courses_df: Optional[pd.DataFrame] = None

    def fit(self, courses_df: pd.DataFrame):
        """
        Huấn luyện recommender trên dữ liệu khóa học:
        - Loại bỏ trùng lặp theo course_code
        - Kết hợp và tiền xử lý văn bản
        - Chuẩn hóa mã khóa học
        - Xử lý trường semester (nếu có)
        - Sinh embedding cho từng khóa học
        """
        courses_df = courses_df.drop_duplicates(subset=["course_code"]).reset_index(drop=True)
        courses_df["combined_text"] = courses_df.apply(
            lambda r: build_combined_text(r, self.text_cols), axis=1)
        courses_df["processed_text"] = courses_df["combined_text"].apply(preprocess_text)
        courses_df["course_code"] = courses_df["course_code"].astype(str).str.strip().str.upper()
        if "semester" in courses_df.columns:
            # Lấy số học kỳ, nếu không có thì gán 99
            courses_df["semester"] = (
                courses_df["semester"].astype(str)
                .str.extract(r"(\d+)")
                .astype(float)
                .fillna(99)
                .astype(int)
            )
        # Sinh embedding cho toàn bộ khóa học
        self.course_embeddings = self.model.encode(
            courses_df["processed_text"].tolist(),
            show_progress_bar=True,
            normalize_embeddings=True
        ).astype("float32")
        self.course_ids = courses_df["course_code"].tolist()
        self.courses_df = courses_df

    def embed_text(self, text: str) -> np.ndarray:
        """
        Sinh embedding cho một đoạn văn bản đầu vào
        """
        return self.model.encode([preprocess_text(text)],
                                 normalize_embeddings=True).astype("float32")[0]

    def recommend(self, target_emb: np.ndarray,
                  exclude_codes: List[str] = None,
                  top_k: int = 10) -> List[Tuple[str, float]]:
        """
        Gợi ý các khóa học tương tự nhất với embedding đầu vào:
        - Loại trừ các mã khóa học trong exclude_codes
        - Sắp xếp theo độ tương đồng cosine
        - Ưu tiên học kỳ nhỏ hơn (semester tăng dần), sau đó đến score giảm dần
        - Trả về top_k kết quả (danh sách tuple: (course_code, score))
        """
        if self.course_embeddings is None or self.course_ids is None or self.courses_df is None:
            raise RuntimeError("Recommender chưa được huấn luyện dữ liệu khóa học")
        exclude_codes = set(str(x).strip().upper() for x in (exclude_codes or []))
        sims = cosine_similarity(target_emb.reshape(1, -1), self.course_embeddings)[0]
        idxs = np.argsort(-sims)
        scores = sims[idxs]
        raw_results = []
        seen = set()
        for i, idx in enumerate(idxs):
            code = self.course_ids[idx]
            code_norm = str(code).strip().upper()
            if code_norm not in seen and code_norm not in exclude_codes:
                raw_results.append((code_norm, float(scores[i])))
                seen.add(code_norm)
        # Ghép thêm thông tin semester để ưu tiên sắp xếp
        df = pd.DataFrame(raw_results, columns=["course_code", "score"])
        df = df.merge(self.courses_df[["course_code", "semester"]], on="course_code", how="left")
        df_sorted = df.sort_values(by=["semester", "score"], ascending=[True, False])
        final_results = list(df_sorted[["course_code", "score"]].head(top_k).itertuples(index=False, name=None))
        return final_results
