Hệ thống khuyến nghị học tập nâng cao chất lượng đào tạo (LMS Recommender System)
📝 Giới thiệuHệ thống được xây dựng nhằm giải quyết vấn đề sinh viên thiếu tài liệu học tập phù hợp, dẫn đến kết quả thi kém hoặc rớt môn. 
Dự án ứng dụng trí tuệ nhân tạo để cá nhân hóa lộ trình học tập, giúp giảm tỷ lệ nợ môn và nâng cao uy tín giáo dục của nhà trường.
Thành tích: Giải Nhất cuộc thi Nghiên cứu khoa học sinh viên cấp trường năm học 2024 - 2025.
Phạm vi áp dụng: Khoa Công nghệ Thông tin – Trường Đại học Công Nghệ Sài Gòn (STU).
🚀 Tính năng chínhCá nhân hóa lộ trình: 
Sử dụng mô hình học máy để phân tích hồ sơ, thói quen và sở thích của từng sinh viên.
Khuyến nghị thông minh: Đề xuất khóa học, tài liệu và giảng viên phù hợp thông qua thuật toán lọc dựa trên nội dung.
Tích hợp LMS: Hiển thị dưới dạng một plugin block trực quan trên hệ thống Moodle.
Lọc điều kiện tiên quyết: Tự động loại trừ môn đã học và kiểm tra các môn tiên quyết trước khi gợi ý.
🛠 Công nghệ sử dụngBackend: Python Flask API.
Frontend: HTML, CSS, JavaScript, React (Moodle Plugin).
Machine Learning: sentence-transformers (DistilBERT), scikit-learn.Database: MySQL (kết nối trực tiếp từ dữ liệu Moodle).
📐 Kiến trúc hệ thốngHệ thống được thiết kế theo kiến trúc 3 lớp:
Giao diện (Frontend): Moodle Plugin Block.
Xử lý trung gian (Backend): Flask API xử lý logic và truy xuất dữ liệu.Engine Khuyến nghị: Tính toán độ tương đồng Cosine giữa hồ sơ sinh viên và môn học.🧠 Thuật toán khuyến nghị
Dự án áp dụng Content-Based Filtering (CBF) với quy trình:
Vector hóa thông tin môn học và sở thích sinh viên bằng SentenceTransformer.
Tính toán độ tương đồng bằng công thức Cosine Similarity
Xếp hạng và trả về danh sách Top-K môn học phù hợp nhất.
👥 Thành viên thực hiện 
Giảng viên hướng dẫn: ThS. Mai Vân Phương Vũ.
Sinh viên thực hiện: Trần Tuấn Anh (Chủ nhiệm đề tài), Mè Thái Huy, Đậu Quốc Khánh, Triệu Kim Long, Trần Nhựt Quang.
© 2026 - Dự án Nghiên cứu Khoa học STU
