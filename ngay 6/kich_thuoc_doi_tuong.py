# coding: utf-8
from scipy.spatial import distance as dist
from imutils import perspective
from imutils import contours
import numpy as np
import argparse
import imutils
import cv2


#Khai báo hàm tìm điểm nằm giữa 2 điểm (trung điểm)
def diem_giua(diem_A, diem_B):
    return (
            (diem_A[0] + diem_B[0]) * 0.5,
            (diem_A[1] + diem_B[1]) * 0.5,
            )

#Xây dựng các tham số tùy chọn và lấy tùy chọn
tuy_chon = argparse.ArgumentParser()
tuy_chon.add_argument(
        '-i',
        '--image',
        required=True,
        help='Đường dẫn đến ảnh đầu vào',
        )
tuy_chon.add_argument(
        '-w',
        '--width',
        type=float,
        required=True,
        help='''Chiều dài của đối tượng nằm ngoài cùng bên trái bức
        ảnh (inches)''',
        )
cac_tuy_chon = vars(tuy_chon.parse_args())

#Tải hình ảnh, chuyển sang đen trắng, làm mịn
hinh_anh = cv2.imread(cac_tuy_chon['image'])
den_trang = cv2.cvtColor(hinh_anh, cv2.COLOR_BGR2GRAY)
den_trang = cv2.GaussianBlur(den_trang, (7, 7), 0)

#Sử dụng bộ phát hiện cạnh, sau đó làm giãn và làm mượt đường viền
#giữa 2 cạnh của đối tượng
canh_phat_hien = cv2.Canny(den_trang, 50, 100)
canh_phat_hien = cv2.dilate(canh_phat_hien, None, iterations=1)
canh_phat_hien = cv2.erode(canh_phat_hien, None, iterations=1)

#Tìm các đường viền nằm trong vùng bao phủ bởi cạnh
