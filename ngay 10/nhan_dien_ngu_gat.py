#coding: utf-8
from scipy.spatial import distance as dist
from imutils.video import VideoStream
from imutils import face_utils
from threading import Thread
import numpy as np
import playsound
import argparse
import imutils
import time
import dlib
import cv2
import os


def chuong_canh_bao(duong_dan):
    #Phát nhạc cảnh báo
    playsound.playsound(duong_dan)

def ti_le_mat(mat):
    #Tính tỉ lệ trên hình của mắt
    #Bằng cách tính khoảng cách theo Euclidean giũa 2 tập tọa độ (x,
    #y) theo chiều ngang của các điểm mốc của mắt
    A = dist.euclidean(mat[1], mat[5])
    B = dist.euclidean(mat[2], mat[4])

    #Tương tự tính khoảng cách Euclidean giữa 2 tập tọa độ (x, y) theo
    #chiều dọc của các điểm mốc của mắt
    C = dist.euclidean(mat[0], mat[3])

    #Tính tỉ lệ khung hình của mắt
    ti_le = (A + B) / (2.0 * C)
    return ti_le

#Xây dựng tham số tùy chọn và lấy tham số tùy chọn
tuy_chon = argparse.ArgumentParser()
tuy_chon.add_argument(
        '-p',
        '--shape-predictor',
        required=True,
        help="Đường dẫn tệp nhận diện điểm mốc",
        )
tuy_chon.add_argument(
        '-a',
        '--alarm',
        type=str,
        default='',
        help='Đường dẫn âm thanh cảnh báo',
        )
tuy_chon.add_argument(
        '-w',
        '--webcam',
        type=int,
        default=0,
        help='Chỉ số của webcam trong hệ thống',
        )
tuy_chon.add_argument(
        '-v',
        '--video',
        type=str,
        help='Đường dẫn tệp video',
        )
cac_tuy_chon = vars(tuy_chon.parse_args())

#Khởi tạo 2 biến, biến tỉ lệ khung hình của mắt để tính số lần nháy
#mắt, 1 biến cho số khung hình liên tiếp mắt nhắm - biến này nếu vượt
#ngưỡng nhất định sẽ được xem là ngủ gật
NGUONG_TI_LE_MAT = 0.3
SO_KHUNG_HINH = 48

#Khởi tạo bộ đếm khung hình cùng với biến trạng thái của âm cảnh báo
BO_DEM = 0
CHUONG_BAO = False

#Khởi tạo bộ nhận diện khuôn mặt (HOG-based) và sau đó tạo ra bộ nhận
#diện các điểm mốc trên mặt
print('[INFO] Tải tệp nhận diện điểm mốc...')
nhan_dien_khuon_mat = dlib.get_frontal_face_detector()

