def dinh_dang_vnd(so_tien):
    """Định dạng số tiền theo VND."""
    return f"{so_tien:,.0f} VND"


def dinh_dang_phan_tram(gia_tri):
    """Định dạng tỉ lệ phần trăm."""
    return f"{gia_tri * 100:.1f}%"


def tom_tat_thong_tin_xe(thong_tin_xe):
    """Tạo chuỗi tóm tắt thông tin xe."""
    hang_xe = thong_tin_xe.get("brand", "").strip()
    dong_xe = thong_tin_xe.get("model", "").strip()
    nam_sx = thong_tin_xe.get("year", "")
    km_da_di = thong_tin_xe.get("km_driven", "")
    nhien_lieu = thong_tin_xe.get("fuel_type", "")
    hop_so = thong_tin_xe.get("transmission", "")

    return f"{hang_xe} {dong_xe} {nam_sx} | {km_da_di} km | {nhien_lieu} | {hop_so}"
