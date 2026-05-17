# ==============================================================================
# PHẦN 3.1: HÀM TÍNH GIAI THỪA
# ==============================================================================
def tinh_giai_thua(n):
    if n < 0:
        return "Không tính được giai thừa cho số âm"
    giai_thua = 1
    for i in range(1, n + 1):
        giai_thua *= i
    return giai_thua


# ==============================================================================
# PHẦN 3.2: HÀM TÍNH GIÁ TRỊ TRUNG BÌNH CỦA DÃY SỐ
# ==============================================================================
def tinh_trung_binh(danh_sach):
    if len(danh_sach) == 0:
        return 0
    return sum(danh_sach) / len(danh_sach)


# ==============================================================================
# PHẦN 3.3: HÀM TÍNH LỢI NHUẬN SAU 12 THÁNG (LÃI KÉP)
# ==============================================================================
def tinh_loi_nhuan_12_thang(so_tien_goc, lai_suat_nam):
    lai_suat_thang = (lai_suat_nam / 100) / 12
    so_tien_cuoi_ky = so_tien_goc * (1 + lai_suat_thang)**12
    loi_nhuan = so_tien_cuoi_ky - so_tien_goc
    return so_tien_cuoi_ky, loi_nhuan


# ==============================================================================
# KHU VỰC CHẠY THỬ NGHIỆM (MAIN)
# ==============================================================================
if __name__ == "__main__":
    print("--- KẾT QUẢ CHẠY BÀI TẬP PYTHON CƠ BẢN --- \n")
    
    # 1. Chạy thử bài 3.1
    n_test = 5
    print(f"[3.1] Giai thừa của {n_test} là: {tinh_giai_thua(n_test)}")
    print("-" * 40)
    
    # 2. Chạy thử bài 3.2
    day_so_test = [10, 20, 30, 40, 50]
    print(f"[3.2] Giá trị trung bình của dãy là: {tinh_trung_binh(day_so_test)}")
    print("-" * 40)
    
    # 3. Chạy thử bài 3.3
    goc_test = 100000000  # 100 triệu VND
    lai_suat_test = 6.0   # 6% / năm
    tong_tien, lai = tinh_loi_nhuan_12_thang(goc_test, lai_suat_test)
    print(f"[3.3] Kết quả tính toán tài chính sau 12 tháng:")
    print(f"      + Tổng số tiền nhận được (Cả gốc lẫn lãi): {tong_tien:,.2f} VND")
    print(f"      + Tiền lãi thuần túy: {lai:,.2f} VND")
    print("=" * 40)