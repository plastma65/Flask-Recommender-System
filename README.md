# Xây dựng hệ thống hỗ trợ nâng cao chất lượng học tập cho sinh viên

Bản hiện tại gồm website **gợi ý giảng viên theo nhu cầu tiếng Việt** và **dự đoán PASS/FAIL**, sử dụng React + Vite + Ant Design và FastAPI. Danh mục gồm 45 giảng viên. Tên kho được giữ để bảo toàn đường dẫn; các script Flask/Moodle ở thư mục gốc thuộc phiên bản trước, xem [README cũ](docs/README_LEGACY.md).

## Phạm vi công khai

Kho chỉ công khai mã nguồn và hướng dẫn của phiên bản hiện tại. **Dữ liệu giảng viên và bốn tệp mô hình không nằm trong bản cập nhật này.** Muốn chạy đầy đủ, cần nhận bộ tài nguyên được phép sử dụng từ nhóm nghiên cứu và đặt đúng vị trí bên dưới. Cài thư viện hoặc tải encoder không thay thế các tài nguyên đó.

```text
Project/Data/teacher_profiles_cleaned.csv
Project/Data/teacher_courses.csv
Project/smart-learning-backend/teacher_svd_model.pkl
Project/smart-learning-backend/teacher_embeddings.pkl
Project/smart-learning-backend/pass_prediction_model.pkl
Project/smart-learning-backend/pass_prediction_encoders.pkl
```

## Cài đặt

Yêu cầu Python **3.11**, Node.js **22.12 trở lên** tương thích Vite, Git và Internet để cài thư viện/tải encoder lần đầu. Windows có thể cần Microsoft C++ Build Tools để biên dịch scikit-surprise. Hai chức năng demo chính không cần PostgreSQL hay Moodle.

Từ PowerShell, bỏ tải checkpoint Git LFS cũ vì bản mới không sử dụng:

```powershell
$env:GIT_LFS_SKIP_SMUDGE="1"
git clone https://github.com/plastma65/Flask-Recommender-System.git
cd Flask-Recommender-System
py -3.11 -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip setuptools wheel
.\.venv\Scripts\python.exe -m pip install numpy==1.26.4
.\.venv\Scripts\python.exe -m pip install --no-build-isolation -r Project\smart-learning-backend\requirements-lock.txt
.\.venv\Scripts\python.exe scripts\download_encoder.py
```

Giữ phiên bản trong `requirements-lock.txt` để tương thích mô hình. Trên Linux/macOS, dùng `.venv/bin/python` và đặt `GIT_LFS_SKIP_SMUDGE=1` trước lệnh clone.

## Chạy website

**Terminal 1**, từ thư mục gốc kho:

```powershell
cd Project\smart-learning-backend
..\..\.venv\Scripts\python.exe -m uvicorn app.main:app --host 127.0.0.1 --port 8000
```

**Terminal 2**, từ thư mục gốc kho:

```powershell
cd Project\smart-learning-frontend
npm ci
npm run dev -- --host 127.0.0.1
```

Mở http://127.0.0.1:5173. API docs: http://127.0.0.1:8000/docs. Dừng bằng Ctrl+C ở mỗi terminal. Hai chức năng chính không cần `.env`; có mẫu `.env.example` nếu cần tùy chỉnh.

## Demo

- Nhập “Tôi muốn học lập trình thiết bị di động.” để kiểm tra gợi ý phù hợp môn học. Có thể nhập lại không dấu để kiểm tra đối chiếu tên môn. Kết quả phụ thuộc bộ tài nguyên bàn giao riêng.
- PASS: GPA từ 2,5 đến 3,19; tự học 0–5 giờ/tuần; 0 môn trượt; điểm danh trên 90%. PASS khoảng 96,7%.
- FAIL: GPA từ 2,0 đến 2,49; tự học 0–5 giờ/tuần; từ 4 môn trượt; điểm danh 70–90%. FAIL khoảng 84,3%.

Ví dụ chỉ minh họa chức năng, không đo độ chính xác tổng quát. Thứ tự giảng viên không phải đánh giá chất lượng giảng dạy; xác suất không bảo đảm kết quả học tập.

## Kiểm tra

Sau khi nhận bộ tài nguyên riêng và tải encoder, từ thư mục gốc:

```powershell
.\.venv\Scripts\python.exe -m pip install httpx
.\.venv\Scripts\python.exe scripts\check_demo.py
cd Project\smart-learning-frontend
npm run build
```

Nếu log báo mô hình chưa sẵn sàng, kiểm tra bốn tệp `.pkl` trong backend và chạy lại bước tải encoder. Nếu cổng đang bận, dừng phiên demo cũ trước khi chạy lại.

## Phạm vi bàn giao

- `Project/smart-learning-frontend`: giao diện, cấu hình, khóa phiên bản npm.
- `Project/smart-learning-backend`: API, cấu hình mẫu và thư viện Python.
- `scripts`: tải encoder và kiểm tra demo; `docs`: ghi chú bàn giao dữ liệu.

Bản cập nhật không công khai dữ liệu, mô hình hay khảo sát sinh viên của phiên bản hiện tại. Xem [tài nguyên cần nhận riêng](docs/DATA_AND_MODELS.md). Các API môn học, hồ sơ mẫu và theo dõi còn là khung thử nghiệm, chưa phải chức năng hoàn chỉnh.

## Nhóm thực hiện

Giảng viên hướng dẫn: **ThS. Mai Vân Phương Vũ**.

- Trần Tuấn Anh — chủ nhiệm đề tài
- Mè Thái Huy
- Giang Ca Diếp
- Triệu Kim Long

Trường Đại học Công nghệ Sài Gòn. Mã nguồn theo [MIT](LICENSE); dữ liệu và mô hình bên thứ ba theo điều kiện nguồn tương ứng.
