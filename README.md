# HỆ THỐNG GIÁM SÁT MỨC ĐỘ TẬP TRUNG CỦA SINH VIÊN SỬ DỤNG THỊ GIÁC MÁY TÍNH

> Đồ án môn học **Phân tích và Nhận dạng mẫu**  
> Khoa Công nghệ Thông tin – Trường Đại học Sài Gòn

---

# Giới thiệu

Hệ thống Giám sát mức độ tập trung của sinh viên là một ứng dụng sử dụng **Thị giác máy tính (Computer Vision)** nhằm theo dõi và đánh giá trạng thái tập trung của sinh viên trong quá trình học trực tuyến hoặc làm bài kiểm tra.

Hệ thống sử dụng camera để thu nhận hình ảnh theo thời gian thực, sau đó áp dụng các thuật toán phát hiện khuôn mặt, theo dõi hướng nhìn, ước lượng tư thế đầu và phát hiện điện thoại để xác định mức độ tập trung của người học.

Đây là đề tài được xây dựng phục vụ môn học **Phân tích và Nhận dạng mẫu**, trong đó tập trung vào quy trình:

- Trích xuất đặc trưng (Feature Extraction)
- Phân loại mẫu (Pattern Classification)
- Đánh giá hiệu năng hệ thống (Performance Evaluation)

---

# Mục tiêu

- Xây dựng hệ thống theo dõi sinh viên theo thời gian thực.
- Phát hiện sinh viên có xuất hiện trước camera hay không.
- Xác định hướng nhìn của mắt.
- Ước lượng hướng quay của đầu.
- Phát hiện việc sử dụng điện thoại.
- Tính toán mức độ tập trung (Focus Score).
- Ghi nhận các hành vi vi phạm.
- Xuất báo cáo sau mỗi phiên làm việc.

---

# Chức năng chính

Hệ thống hiện hỗ trợ các chức năng sau:

- Phát hiện khuôn mặt (Face Detection)
- Theo dõi hướng nhìn của mắt (Eye Gaze Tracking)
- Ước lượng tư thế đầu (Head Pose Estimation)
- Phát hiện điện thoại bằng YOLOv8
- Phát hiện người
- Phân tích trạng thái tập trung
- Tính điểm tập trung
- Ghi nhận vi phạm
- Xuất báo cáo CSV
- Giao diện Dashboard theo thời gian thực

---

# Quy trình nhận dạng mẫu

Hệ thống được xây dựng theo quy trình chuẩn của môn **Phân tích và Nhận dạng mẫu**.

```

Camera
↓
Thu nhận ảnh
↓
Tiền xử lý
↓
Phát hiện khuôn mặt
↓
Trích xuất đặc trưng
↓
Phân loại mẫu
↓
Đánh giá mức độ tập trung
↓
Hiển thị kết quả và xuất báo cáo

```

---

# Kiến trúc hệ thống

```

Webcam
│
▼
Face Detection
│
▼
MediaPipe Face Mesh
│
├───────────────┐
│ │
▼ ▼
Eye Gaze Head Pose
│ │
└───────────────┘
│
▼
Feature Extraction
│
▼
Pattern Classification
│
▼
Focus Analyzer
│
▼
Dashboard
│
▼
Report Generator

```

---

# Trích xuất đặc trưng (Feature Extraction)

Trong mỗi khung hình, hệ thống trích xuất các đặc trưng sau:

| Đặc trưng | Ý nghĩa |
|-----------|----------|
| Face Presence | Có khuôn mặt hay không |
| Gaze Direction | Hướng nhìn của mắt |
| Head Direction | Hướng quay của đầu |
| Eyes Closed | Trạng thái nhắm mắt |
| Phone Detected | Có phát hiện điện thoại |
| Person Detected | Có phát hiện người |

Các đặc trưng này được tổng hợp thành **vector đặc trưng** và đưa vào bộ phân loại.

---

# Phân loại mẫu (Pattern Classification)

Hệ thống sử dụng phương pháp **Rule-based Classification**.

Các luật phân loại được xây dựng như sau:

```

Nếu không có khuôn mặt
→ No Face

Nếu phát hiện điện thoại
→ Phone Detected

Nếu nhắm mắt
→ Eyes Closed

Nếu nhìn sang trái
→ Looking Left

Nếu nhìn sang phải
→ Looking Right

Nếu nhìn xuống
→ Looking Down

Ngược lại
→ Focused

```

---

# Cấu trúc thư mục

```

Student-Focus-Monitoring-System

│

├── app.py

├── modules/

│ ├── face_detector.py

│ ├── gaze_tracker.py

│ ├── head_pose.py

│ ├── phone_detector.py

│ ├── person_detector.py

│ ├── feature_extractor.py

│ ├── focus_analyzer.py

│ ├── report_generator.py

│ └── violation_logger.py

│

├── reports/

├── evaluation/

├── assets/

├── README.md

├── requirements.txt

└── yolov8n.pt

```

---

# Công nghệ sử dụng

- Python
- OpenCV
- MediaPipe
- YOLOv8
- NumPy
- Pandas
- Matplotlib

---

# Hướng dẫn cài đặt

### Bước 1: Tạo môi trường ảo

```bash
python -m venv venv
```

### Bước 2: Kích hoạt môi trường

Windows

```bash
venv\Scripts\activate
```

### Bước 3: Cài đặt thư viện

```bash
pip install -r requirements.txt
```

---

# Chạy chương trình

```bash
python app.py
```

Nhấn **ESC** để kết thúc chương trình.

---

# Kết quả đầu ra

Sau khi chạy, hệ thống cung cấp:

- Giao diện Dashboard
- Điểm tập trung
- Hướng nhìn
- Hướng đầu
- Trạng thái sinh viên
- Danh sách vi phạm
- Báo cáo CSV

---

# Đánh giá hệ thống

Hệ thống được đánh giá dựa trên các chỉ số:

- Accuracy
- Precision
- Recall
- F1-score
- Confusion Matrix

Kết quả đánh giá được thực hiện trên bộ dữ liệu kiểm thử do nhóm xây dựng.

---

# Hướng phát triển

Trong tương lai, hệ thống có thể được mở rộng theo các hướng:

- Phân loại bằng CNN
- Nhận diện khuôn mặt
- Theo dõi nhiều sinh viên
- Phân tích hành vi theo chuỗi thời gian (LSTM)
- Dashboard Web
- Lưu dữ liệu vào cơ sở dữ liệu

---

# Thông tin đồ án

**Tên đề tài**

Hệ thống giám sát mức độ tập trung của sinh viên sử dụng Thị giác máy tính

**Môn học**

Phân tích và Nhận dạng mẫu

**Đơn vị thực hiện**

Khoa Công nghệ Thông tin

Trường Đại học Sài Gòn

---

# Giấy phép

Đồ án được phát triển phục vụ mục đích học tập và nghiên cứu.