# !/usr/bin/env python
# -*-coding:utf-8 -*-

"""
# File       : run_IR.py
# Time       ：2023/7/28 16:21
# Author     ：Zheng Youcai[youcaizheng@foxmail.com]
# version    ：python 3.10
# Description：有带畸变的图片，并处理（手动PS畸变）
"""
import os

from calibrate_helper import Calibrator


def main():
    # img_dir = "./pic/ir"    # 横向径向都处理
    img_dir = "./pic/ir_radial"     # 仅径向
    shape_inner_corner = (6, 9)
    size_grid = 0.02
    calibrator = Calibrator(img_dir, shape_inner_corner, size_grid)
    # 求参
    mtx, dist = calibrator.calibrate_camera()
    # 校正
    # save_dir = "./pic/IR_dedistortion"
    save_dir = "./pic/IR_radial_dedistortion"
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)
    calibrator.dedistortion(save_dir)


if __name__ == '__main__':
    main()
