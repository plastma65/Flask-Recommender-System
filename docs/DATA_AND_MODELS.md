# Tài nguyên cần bàn giao riêng

Theo phạm vi công khai đã lựa chọn, bản cập nhật này chỉ có mã nguồn và hướng dẫn. Không tải lên danh mục giảng viên, liên kết môn học, các tệp mô hình, thông tin refit hay khảo sát sinh viên của phiên bản hiện tại.

Để chạy hai chức năng chính, người nhận cần bộ tài nguyên riêng từ nhóm nghiên cứu: `teacher_profiles_cleaned.csv`, `teacher_courses.csv` trong `Project/Data`; `teacher_svd_model.pkl`, `teacher_embeddings.pkl`, `pass_prediction_model.pkl`, `pass_prediction_encoders.pkl` trong `Project/smart-learning-backend`. Không có các tệp này thì API mô hình chưa sẵn sàng, dù cài đặt thư viện thành công.

Encoder công khai `keepitreal/vietnamese-sbert` tải bằng `scripts/download_encoder.py`. Đây là tài nguyên bổ sung, không thay thế bốn tệp mô hình nội bộ. Chỉ nạp pickle từ nguồn tin cậy.

Mã và các CSV/checkpoint ở thư mục gốc thuộc phiên bản Flask trước đây, đã có trên kho trước lần cập nhật này. Chúng được giữ nguyên, không được ứng dụng hiện tại sử dụng và không thay thế bộ dữ liệu của báo cáo mới. Không dùng chúng để suy diễn kết quả của phiên bản hiện tại.

Không thể tự tái lập toàn bộ huấn luyện và đánh giá nghiên cứu chỉ từ mã nguồn công khai này. Việc tiếp cận dữ liệu và mô hình riêng cần tuân theo phạm vi bàn giao của nhóm nghiên cứu.
