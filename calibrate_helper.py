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
        self.mat_intri = None   # 内参矩阵
        self.coff_dis = None    # 镜头畸变参数

        # 世界坐标
        w, h = shape_inner_corner
        # 世界空间中角点的坐标, int形式
        objp = np.zeros((w * h, 3), np.float32)
        objp[0, :, :2] = np.mgrid[0:w, 0:h].T.reshape(-1, 2)
        # 世界角点坐标
        self.cp_world = objp * size_grid

        # 图片
        self.img_paths = []
        for extension in ["jpg", "png", "jpeg"]:
            self.img_paths += glob.glob(os.path.join(img_dir, "*.{}".format(extension)))
        assert len(self.img_paths), "no images for calibration found!"

    def calibrate_camera(self):


