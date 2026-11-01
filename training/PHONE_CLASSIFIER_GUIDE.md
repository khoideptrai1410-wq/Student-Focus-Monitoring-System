# Quy trình huấn luyện và đánh giá nhận diện điện thoại

## Sơ đồ luồng

```text
Thu thập ảnh đa dạng
        |
        v
Gán nhãn đúng: PHONE / NOT_PHONE
        |
        v
Kiểm tra ảnh, sửa nhãn sai và cân bằng dữ liệu
        |
        v
Huấn luyện CNN -> models/phone_classifier.pth
        |
        v
Thử trong một buổi camera mới, với vật thể/bối cảnh khác
        |
        +---- Bỏ sót điện thoại? ---> Thêm mẫu PHONE khó -> Huấn luyện lại
        |
        +---- Báo nhầm vật khác? ---> Thêm mẫu NOT_PHONE gây nhầm -> Huấn luyện lại
        |
        v
Đạt yêu cầu -> Khởi động lại app để nạp checkpoint
```

## 0. Chuẩn bị môi trường (chỉ làm một lần)

Mở PowerShell tại thư mục gốc dự án. Dự án hiện dùng Python 3.12.10 trong
`.venv`. Nếu chưa có môi trường này, tạo bằng Python 3.12:

```powershell
py -3.12 -m venv .venv
```

Cài các thư viện (chạy lại lệnh này sau khi thay đổi `requirements.txt`):

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

Các lệnh bên dưới dùng trực tiếp Python trong `.venv`, không cần kích hoạt
venv. Lần đầu tải trọng số pretrained có thể cần kết nối Internet.

> Quy trình này huấn luyện CNN MobileNetV3 để **xác minh ứng viên điện thoại**
> do YOLO đề xuất; không huấn luyện lại YOLO, nhận diện khuôn mặt hay theo dõi
> hướng nhìn.

## 1. Thu thập ảnh huấn luyện

Mở PowerShell tại thư mục gốc dự án:

```powershell
.\.venv\Scripts\python.exe training\collect_phone_candidates.py
```

Trong cửa sổ camera:

- Nhấn `P` khi vùng đang khoanh là điện thoại.
- Nhấn `N` khi vùng đang khoanh không phải điện thoại.
- Nhấn `S` để chọn ứng viên tiếp theo nếu có nhiều vùng.
- Nhấn `Q` để kết thúc.

Thử cầm điện thoại trên tay nhiều lần, thay đổi góc, khoảng cách, ánh sáng và
nền. Lớp `NOT_PHONE` nên có chuột, chai nước, thẻ xe và các vật thường bị nhầm.
Chỉ gán nhãn khi chắc chắn vùng khoanh và nhãn là đúng.

Ảnh huấn luyện được lưu tại:

```text
data\phone_classifier\raw\PHONE
data\phone_classifier\raw\NOT_PHONE
```

Trước khi huấn luyện, mở và kiểm tra ảnh trong hai thư mục này. Xóa hoặc di
chuyển ảnh gán nhãn sai và ảnh trùng lặp; cố gắng giữ số lượng và độ đa dạng
tương đối cân bằng giữa hai lớp. Mỗi lớp cần ít nhất 5 ảnh để script chạy,
nhưng đây chỉ là mức tối thiểu, không phải bộ dữ liệu đủ tốt.

Hiện bộ dữ liệu của dự án có hơn 63 ảnh `PHONE` và hơn 60 ảnh `NOT_PHONE`.
Số lượng gần cân bằng nhưng chưa chứng minh độ chính xác cao: cần ưu tiên ảnh
đúng nhãn, nhiều điện thoại/vật thể, góc, khoảng cách, ánh sáng và buổi thu
khác nhau. Tránh chỉ chụp liên tiếp cùng một cảnh vì ảnh gần giống nhau có thể bị chia sang cả train và validation.

## 2. Huấn luyện

```powershell
.\.venv\Scripts\python.exe training\train_phone_classifier.py
```

Mặc định script dùng MobileNetV3 pretrained, 15 epoch, batch size 16; tự chọn
CUDA nếu PyTorch nhận GPU, nếu không thì chạy CPU. Đây là thiết lập cân bằng tốc
độ và chất lượng, không có thiết lập nào bảo đảm chính xác nhất cho mọi camera.
Đừng thêm `--no-pretrained` trừ khi không thể tải trọng số pretrained.

Script in accuracy train/validation theo từng epoch và lưu checkpoint tốt nhất
vào `models\phone_classifier.pth`. Chờ thông báo `Training finished` trước khi
chạy app. Không đánh giá mô hình chỉ bằng validation accuracy: ảnh gần giống
nhau có thể bị chia sang cả train và validation.

## 3. Đánh giá bằng buổi thử riêng

1. Dùng app trong một buổi camera mới, với góc, nền, ánh sáng và vật thể khác
   bộ huấn luyện. Không chép ảnh đánh giá vào `raw` trước khi đánh giá xong.
2. Thử cả hai tình huống: cầm điện thoại và cầm từng vật không phải điện thoại.
   Ghi lại mỗi lần app phát hiện đúng, bỏ sót điện thoại hoặc báo nhầm vật khác.
3. Tính:
   - `Precision = TP / (TP + FP)`: trong các lần app báo điện thoại, bao nhiêu lần đúng.
   - `Recall = TP / (TP + FN)`: trong các lần cầm điện thoại, bao nhiêu lần được phát hiện.
   - `TP`: phát hiện đúng điện thoại; `FP`: báo nhầm vật khác; `FN`: bỏ sót điện thoại.
4. Nếu bỏ sót, thêm các mẫu điện thoại tương tự vào `PHONE`. Nếu báo nhầm, thêm chính các vật gây nhầm vào `NOT_PHONE`. Chạy lại lệnh huấn luyện ở bước 2 và đánh giá trên một buổi mới.

Để so sánh các lần cập nhật, ghi lại TP, FP, FN và thời gian từ lúc đưa điện
thoại vào khung hình đến lúc hiện khung bao. Chọn mức chấp nhận trước khi thử; đừng hạ ngưỡng chỉ để tăng recall nếu việc đó làm báo nhầm tăng không chấp nhận được.

Nếu YOLO không khoanh điện thoại thì CNN không có vùng để phân loại; đây là lỗi đề xuất của YOLO, không phải lỗi CNN. Nếu YOLO khoanh được nhưng CNN từ chối,
kiểm tra nhãn và mẫu huấn luyện CNN.

## 4. Dùng checkpoint mới trong app

Sau khi đánh giá đạt yêu cầu, đóng app nếu đang mở và khởi động lại:

```powershell
.\.venv\Scripts\python.exe app.py
```

Ứng dụng sẽ nạp `models\phone_classifier.pth` khi khởi động.
