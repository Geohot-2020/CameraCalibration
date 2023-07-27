import cv2
import glob
import numpy as np


# 定义棋盘尺寸
CHECKERBOARD = (6, 9)
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

# 3D点阵
objPoints = []
# 2D点阵
imgPoints = []

# 世界坐标
objp = np.zeros((1, CHECKERBOARD[0]*CHECKERBOARD[1], 3), np.float32)
objp[0, :, :2] = np.mgrid[0:CHECKERBOARD[0], 0:CHECKERBOARD[1]].T.reshape(-1, 2)
prev_img_shape = None

# glob拉所有图片
images = glob.glob('./pic/rgb2ir/*.jpg')
for fname in images:
    img = cv2.imread(fname)
    # 转灰度
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # 用opencv函数找所需数量的角,查找成功ret返回true
    ret, corners = cv2.findChessboardCorners(gray, CHECKERBOARD, cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_FAST_CHECK + cv2.CALIB_CB_NORMALIZE_IMAGE)
    # 如果检测到所需数量的角点，我们将细化像素坐标并将其显示在棋盘图像上
    if ret:
        # 3D点阵
        objPoints.append(objp)

        # 细化像素坐标
        corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
        # 2D点阵
        imgPoints.append(corners2)

        # 画图显示
        img = cv2.drawChessboardCorners(img, CHECKERBOARD, corners2, ret)

    # 显示
    cv2.namedWindow('img', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('img', 800, 800)
    cv2.moveWindow("img", 1000, 300)
    cv2.imshow('img', img)
    cv2.waitKey(0)

cv2.destroyAllWindows()

h, w = img.shape[:2]

# 标定
# 通过传递已知3D点（objpoints）的值和检测到的角点（imgpoints）的相应像素坐标来执行相机校准
ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objPoints, imgPoints, gray.shape[::-1], None, None)

# 内参矩阵
print("mtx：")
print(mtx)
# 镜头畸变参数
print("dist: ")
print(dist)
# 旋转向量
print("rvecs: ")
print(tvecs)
# 平移向量
print("tvecs : ")
print(tvecs)

