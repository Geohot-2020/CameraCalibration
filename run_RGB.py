# !/usr/bin/env python
# -*-coding:utf-8 -*-

"""
# File       : run_RGB.py
# Time       ：2023/7/27 22:15
# Author     ：Zheng Youcai[youcaizheng@foxmail.com]
# version    ：python 3.10
# Description：不带畸变的图像，求内参矩阵、镜头畸变、平移向量、旋转向量
"""
import os

from calibrate_helper import Calibrator


def main():
    img_dir = "pic/rgb"
    shape_inner_corner = (6, 9)
    size_grid = 0.02
    calibrator = Calibrator(img_dir, shape_inner_corner, size_grid)
    mtx, dist = calibrator.calibrate_camera()


if __name__ == '__main__':
    main()
