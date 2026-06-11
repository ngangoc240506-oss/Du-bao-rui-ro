# 🛡️ Ứng Dụng Web Phát Hiện Giao Dịch Gian Lận Tài Chính (Streamlit)

Ứng dụng web này được chuyển đổi tự động từ quy trình phân tích và huấn luyện mô hình học máy (Machine Learning) trong tệp Jupyter Notebook `phat_hien_giao_dich_gian_lan.ipynb`. Hệ thống hỗ trợ người dùng tải dữ liệu mẫu, khám phá phân phối, cấu hình trực quan tham số trực tiếp và thực hiện dự đoán rủi ro gian lận trực tuyến.

## 🤖 Thuật Toán Sử Dụng
- **Mô hình chính:** `Random Forest Classifier` (Họ thuật toán học có giám sát - Phân loại nhị phân).
- **Mục tiêu:** Dự đoán khả năng xảy ra gian lận hoặc vỡ nợ (nhãn cột `default`: `0` là bình thường, `1` là gian lận).

## 📊 Cấu Trúc File Dữ Liệu Đầu Vào (Yêu Cầu)
Để ứng dụng hoạt động chính xác không gây lỗi hệ thống, tệp tải lên cần tuân thủ cấu trúc định dạng sau:
- **Biến độc lập/đặc trưng đầu vào ($X$):** Gồm 14 cột định lượng liên tục từ `X_1`, `X_2`, `X_3`, ..., `X_14`.
- **Biến mục tiêu phụ thuộc ($y$):** Cột mang tên `default` chứa giá trị nhị phân `[0, 1]`. *(Lưu ý: Đối với chế độ dự báo hàng loạt ở Tab 4, không bắt buộc phải có cột nhãn `default` này).*

---

## 🛠️ Hướng Dẫn Cài Đặt Và Chạy Ứng Dụng

### Bước 1: Khởi tạo và kích hoạt môi trường ảo (Khuyến nghị)
```bash
# Trên Windows
python -m venv venv
.\venv\Scripts\activate

# Trên macOS/Linux
python3 -m venv venv
source venv/bin/activate
