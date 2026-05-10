from PIL import Image, ImageDraw

LOAI_NHIEN_LIEU = [
    "Petrol",
    "Diesel",
    "Hybrid",
    "Electric",
    "CNG",
    "LPG",
    "Other",
]

HOP_SO = [
    "Automatic",
    "Manual",
    "CVT",
    "Semi-Automatic",
]

THONG_TIN_XE_MAC_DINH = {
    "brand": "Toyota",
    "model": "Vios",
    "year": 2018,
    "num_seats": 5,
    "km_driven": 45000,
    "fuel_type": "Petrol",
    "transmission": "Automatic",
    "owner_count": 1,
    "fuel_consumption": 18.0,
    "engine_cc": 1500.0,
    "max_power": 107.0,
}

THONG_TIN_XE_MAU = {
    "brand": "Honda",
    "model": "Civic",
    "year": 2020,
    "num_seats": 5,
    "km_driven": 32000,
    "fuel_type": "Petrol",
    "transmission": "Automatic",
    "owner_count": 1,
    "fuel_consumption": 16.0,
    "engine_cc": 1800.0,
    "max_power": 140.0,
}


def tao_anh_mau():
    """Tạo ảnh mẫu để demo nhanh."""
    nhan = [
        ("front.jpg", "Front View"),
        ("rear.jpg", "Rear View"),
        ("left.jpg", "Left Side"),
        ("right.jpg", "Right Side"),
    ]

    danh_sach_anh = []
    for ten_tap, ten_nhan in nhan:
        anh = Image.new("RGB", (960, 540), (245, 246, 248))
        ve = ImageDraw.Draw(anh)
        ve.rectangle([40, 60, 920, 480], outline=(120, 140, 160), width=4)
        ve.text((60, 80), "Sample Image", fill=(60, 70, 80))
        ve.text((60, 120), ten_nhan, fill=(80, 90, 100))
        danh_sach_anh.append({"name": ten_tap, "image": anh})

    return danh_sach_anh
