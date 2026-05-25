import streamlit as st
import pandas as pd
import os
from dotenv import load_dotenv
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import IsolationForest

# Nạp cấu hình bảo mật từ tệp ẩn .env (nếu có)
load_dotenv()

# 1. Cấu hình tiêu đề trang web Dashboard
st.set_page_config(page_title="Hệ thống Cảnh báo Gian lận", layout="wide")
st.title("📊 HỆ THỐNG MÀNG LỌC VÀ CẢNH BÁO GIAO DỊCH BẤT THƯỜNG")
st.subheader("Dành cho bộ phận Kiểm soát nội bộ và Kế toán trưởng")

# 2. Đọc dữ liệu tài chính hiệu năng cao bằng đường dẫn bảo mật
@st.cache_data
def load_data():
    # Lấy đường dẫn từ biến môi trường, nếu không có sẽ dùng file mặc định trong thư mục
    data_path = os.getenv("FINANCIAL_DATA_PATH", "financial_anomaly_data.csv")
    df = pd.read_csv(data_path)
    df.columns = df.columns.str.strip().str.lower()
    return df

try:
    df = load_data()
    
    # 3. Chạy mô hình AI phát hiện gian lận tài chính
    X = df.drop(columns=['class']) if 'class' in df.columns else df
    
    # LỌC SẠCH DỮ LIỆU: Chỉ giữ lại thuộc tính số để triệt tiêu hoàn toàn lỗi chuyển đổi chuỗi ký tự/ngày tháng
    X_numeric = X.select_dtypes(include=['number'])
    
    # Chuẩn hóa thang đo số liệu về dạng phân phối chuẩn Z-score
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_numeric)
    
    # Kích hoạt mô hình Rừng cô lập với tỷ lệ bao phủ rủi ro 0.5%
    model = IsolationForest(contamination=0.005, random_state=42)
    df['prediction'] = model.fit_predict(X_scaled)
    
    # Trích xuất danh sách các ca nghi vấn rủi ro (AI đánh dấu bằng nhãn -1)
    danh_sach_ghi_van = df[df['prediction'] == -1].drop(columns=['prediction'])

    # 4. Thiết kế các thẻ số liệu trực quan trên giao diện (Metrics Dashboard)
    col1, col2, col3 = st.columns(3)
    col1.metric("Tổng số giao dịch đã quét", f"{len(df):,}")
    col2.metric("Số ca phát hiện bất thường", f"{len(danh_sach_ghi_van):,}", delta="Cần hậu kiểm ngay", delta_color="inverse")
    col3.metric("Tỷ lệ rủi ro giả định", "0.5 %")

    st.markdown("---")
    
    # 5. Khu vực tương tác và hiển thị danh sách cảnh báo đỏ
    st.error("🚨 DANH SÁCH CÁC GIAO DỊCH CÓ DẤU HIỆU GIAN LẬN (CẦN KIỂM TRA CHỨNG TỪ NGAY)")
    
    # Bộ lọc tìm kiếm nhanh trực tiếp trên giao diện web
    search_query = st.text_input("🔍 Tìm kiếm nhanh giao dịch rủi ro (Nhập thông tin tra cứu):")
    if search_query:
        danh_sach_hien_thi = danh_sach_ghi_van[danh_sach_ghi_van.astype(str).apply(
            lambda x: x.str.contains(search_query, case=False)
        ).any(axis=1)]
    else:
        danh_sach_hien_thi = danh_sach_ghi_van

    # Hiển thị bảng dữ liệu tương tác động, hỗ trợ kéo cuộn mượt mờ
    st.dataframe(danh_sach_hien_thi, use_container_width=True)
    
    # Nút bấm xuất dữ liệu nghi vấn phục vụ công tác thanh tra kế toán
    st.download_button(
        label="📥 Xuất danh sách nghi vấn rủi ro (File CSV)",
        data=danh_sach_hien_thi.to_csv(index=False).encode('utf-8'),
        file_name='danh_sach_giao_dich_rui_ro.csv',
        mime='text/csv',
    )

except FileNotFoundError:
    st.warning("Hệ thống không tìm thấy tệp dữ liệu 'financial_anomaly_data.csv'. Vui lòng kiểm tra lại thư mục!")
except Exception as e:
    st.error(f"Hệ thống phát hiện lỗi vận hành cục bộ: {e}")