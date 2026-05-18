import pandas as pd
import numpy as np

# ==============================================================================
# PHẦN 1 & 2: PHÂN TÍCH KHÁM PHÁ DỮ LIỆU (EDA) & BÁO CÁO KẾT QUẢ
# ==============================================================================
file_name = "financial_anomaly_data.csv"

try:
    # Đọc dữ liệu tài chính
    df = pd.read_csv(file_name)
    print("🎉 TẢI DỮ LIỆU THÀNH CÔNG!\n")
except FileNotFoundError:
    print(f"❌ LỖI: Không tìm thấy file '{file_name}' trong thư mục hiện tại.")
    exit()

print("=" * 60)
print("BÁO CÁO PHÂN TÍCH HỆ THỐNG GIAO DỊCH TÀI CHÍNH")
print("=" * 60)

# 1. Kiểm tra cấu trúc Schema và các cột thực tế
print(f"🔹 Kích thước bộ dữ liệu: {df.shape[0]:,} dòng và {df.shape[1]} cột.")
print("\n🔹 Danh sách các cột thực tế trong file của bạn:")
print(df.columns.tolist())

# TỰ ĐỘNG CHUẨN HÓA: Đổi hết tên cột thành chữ thường và xóa khoảng trắng thừa để tránh lỗi KeyError
df.columns = df.columns.str.strip().str.lower()

# Xác định tên cột mục tiêu dựa trên các tên phổ biến trên Kaggle
target_col = None
for col in ['is_anomaly', 'anomaly', 'is_fraud', 'fraud']:
    if col in df.columns:
        target_col = col
        break

if target_col is None:
    print("\n❌ CẢNH BÁO: Không tìm thấy cột nhãn bất thường. Thầy sẽ lấy cột cuối cùng làm nhãn mục tiêu.")
    target_col = df.columns[-1]

print(f"\n🎯 Cột được xác định làm nhãn phân loại bất thường: '{target_col}'")

# Xác định cột số tiền giao dịch
amount_col = 'amount' if 'amount' in df.columns else None
if amount_col is None:
    # Nếu không thấy chữ 'amount', tìm cột có kiểu dữ liệu số (float/int) trừ cột nhãn
    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if target_col in num_cols:
        num_cols.remove(target_col)
    amount_col = num_cols[0] if num_cols else None

print(f"💰 Cột được xác định là số tiền giao dịch: '{amount_col}'")

# 2. Phân tích tỷ lệ mất cân bằng (Imbalanced Data)
print("\n" + "-" * 50)
print("📊 PHÂN TÍCH TỶ LỆ GIAO DỊCH BẤT THƯỜNG / GIAN LẬN")
print("-" * 50)

counts = df[target_col].value_counts()
percentages = df[target_col].value_counts(normalize=True) * 100

for val in counts.index:
    label_name = "Bất thường (Anomaly/Fraud)" if val == 1 or str(val).lower() == 'true' else "Bình thường (Normal)"
    print(f"   + Nhóm [{val}] {label_name}: {counts[val]:,} dòng ({percentages[val]:.2f}%)")

# 3. Thống kê mô tả số tiền giao dịch
if amount_col:
    print("\n" + "-" * 50)
    print(f"💵 THỐNG KÊ GIÁ TRỊ GIAO DỊCH ({amount_col.upper()}) THEO TỪNG NHÓM")
    print("-" * 50)
    summary = df.groupby(target_col)[amount_col].describe().T
    print(summary)

print("\n" + "=" * 60)
print("📝 KẾT LUẬN CHIẾN LƯỢC AI CỦA GIÁO VIÊN:")
print("=" * 60)
print("1. Đặc trưng: Dữ liệu tài chính có tính chất 'Mất cân bằng dữ liệu' rất rõ rệt.")
print("2. Đánh giá: Khi huấn luyện AI (như Isolation Forest, Random Forest), tuyệt đối không")
print("   dùng Accuracy để đánh giá tổng thể. Phải ưu tiên tối ưu chỉ số Recall và F1-Score")
print("   để đảm bảo không bỏ sót bất kỳ một giao dịch bất thường rủi ro nào.")
print("=" * 60)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Chia dữ liệu (X là các đặc trưng tài chính, y là nhãn gian lận)
X = df.drop(columns=['class'])
y = df['class']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Chuẩn hóa đưa dữ liệu về cùng thang đo
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
from sklearn.ensemble import IsolationForest
from sklearn.metrics import classification_report

# Gọi mô hình Rừng cô lập với tham số tỷ lệ nhiễm bẩn 1%
iso_forest = IsolationForest(contamination=0.01, random_state=42)
iso_forest.fit(X_train_scaled)

# Dự đoán giao dịch bất thường trên tập kiểm thử
y_pred = iso_forest.predict(X_test_scaled)
# Chuyển đổi nhãn dự đoán của Isolation Forest (-1 là bất thường) về dạng nhãn 0 và 1 giống dữ liệu thô
y_pred = [1 if x == -1 else 0 for x in y_pred]

# In báo cáo kết quả Precision, Recall, F1-Score cho giảng viên xem
print("KẾT QUẢ ĐÁNH GIÁ MÔ HÌNH AI:")
print(classification_report(y_test, y_pred))