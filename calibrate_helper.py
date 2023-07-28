# !/usr/bin/env python
# -*-coding:utf-8 -*-

"""
# File       : calibrate_helper.py
# Time       ：2023/7/27 15:14
# Author     ：Zheng Youcai[youcaizheng@foxmail.com]
# version    ：python 3.10
# Description：张正友标定法
"""
import cv2
import os
import glob
import numpy as np


class Calibrator(object):
    def __init__(self, img_dir, shape_inner_corner, size_grid, visualization=True):
        """
        :param img_dir: 图片存储路径, str
        :param shape_inner_corner: 内角形状， array of int, (h, w)
        :param size_grid: 网格实际大小, float
        :param visualization: 可视化选项， bool
        """
        self.img_dir = img_dir
        self.shape_inner_corner = shape_inner_corner
        self.size_grid = size_grid
        self.visualization = visualization
        self.mat_intri = None  # 内参矩阵
        self.coff_dis = None  # 镜头畸变参数

        # 世界坐标
        w, h = shape_inner_corner
        # 世界空间中角点的坐标, int形式
        objp = np.zeros((w * h, 3), np.float32)
        objp[:, :2] = np.mgrid[0:w, 0:h].T.reshape(-1, 2)
        # 世界角点坐标
        self.cp_world = objp * size_grid

        # 图片
        self.img_paths = []
        for extension in ["jpg", "png", "jpeg"]:
            self.img_paths += glob.glob(os.path.join(img_dir, "*.{}".format(extension)))
        assert len(self.img_paths), "no images for calibration found!"

    def calibrate_camera(self):
        w, h = self.shape_inner_corner
        # 设置角点查找的终止条件。在这里使用了一个迭代次数和一个误差阈值的组合，即当迭代次数达到30次或误差小于0.001时，停止角点查找
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
        # 3D点阵
        points_world = []
        # 2D像素点阵
        points_pixel = []
        for img_path in self.img_paths:
            img = cv2.imread(img_path)
            # 2gray
            gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            # 查找所需数量的角
            ret, cp_img = cv2.findChessboardCorners(gray_img, (w, h), None)
            # 查找成功，细化像素坐标
            if ret:
                # 添加坐标
                points_world.append(self.cp_world)
                points_pixel.append(cp_img)
                # 可视化
                if self.visualization:
                    cv2.drawChessboardCorners(img, (w, h), cp_img, ret)

                    cv2.namedWindow('FoundCorners', cv2.WINDOW_NORMAL)
                    cv2.resizeWindow('FoundCorners', 800, 800)
                    cv2.moveWindow('FoundCorners', 1000, 300)
                    cv2.imshow('FoundCorners', img)
                    cv2.waitKey(500)

        # 给参数
        ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(points_world, points_pixel, gray_img.shape[::-1], None, None)

        # 打印参数
        print("内参矩阵: \n{}".format(mtx))
        print("镜头畸变: \n{}".format(dist))
        print("旋转向量: \n{}".format(rvecs))
        print("平移向量: \n{}".format(tvecs))

        # 计算重投影误差：
        # 为每个图像计算重投影误差，即将世界坐标点重新投影到图像上，并计算实际像素坐标和重投影的像素坐标之间的距离。
        # 最后计算平均重投影误差，并输出结果。
        total_error = 0
        for i in range(len(points_world)):
            points_pixel_repro, _ = cv2.projectPoints(points_world[i], rvecs[i], tvecs[i], mtx, dist)
            error = cv2.norm(points_pixel[i], points_pixel_repro, cv2.NORM_L2) / len(points_pixel_repro)
            total_error += error
        print("Average error of reproject: {}".format(total_error / len(points_world)))

        self.mat_intri = mtx
        self.coff_dis = dist
        return mtx, dist

    # 含畸变并校正
    def dedistortion(self, save_dir):
        # 确保获取内参矩阵和畸变参数
        if self.mat_intri is None:
            assert self.coff_dis is None
            self.calibrate_camera()

        w, h = self.shape_inner_corner
        for img_path in self.img_paths:
            _, img_name = os.path.split(img_path)
            img = cv2.imread(img_path)
            # 计算一个优化的相机投影矩阵 newcameramtx 和一个感兴趣区域 roi，用于后续的校正操作
            newcameramtx, roi = cv2.getOptimalNewCameraMatrix(self.mat_intri, self.coff_dis, (w, h), 0, (w, h))
            # dst为校正后的图像
            dst = cv2.undistort(img, self.mat_intri, self.coff_dis, None, newcameramtx)
            cv2.imwrite(os.path.join(save_dir, img_name), dst)
        print("Dedistorted images have been saved to {}".format(save_dir))

