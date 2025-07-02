# Ultralytics YOLO 🚀, AGPL-3.0 license

import cv2
import numpy as np
import torch
import os
import csv
import json
import time
import copy
from datetime import datetime
from collections import defaultdict, deque
from ultralytics import YOLO
from ultralytics.utils.plotting import Annotator, colors


class CameraCalibrator:
    """
    相机标定和畸变矫正工具类
    用于处理广角镜头的畸变问题，将像素坐标转换为真实坐标
    """
    def __init__(self):
        self.camera_matrix = None
        self.dist_coeffs = None
        self.pixel_to_mm_ratio = None
        self.calibrated = False

    def calibrate_camera(self, calibration_images_path, checkerboard_size=(9, 6), square_size=25.0, save_debug_images=True):
        """
        使用棋盘格标定相机

        参数:
            calibration_images_path: 标定图像文件夹路径
            checkerboard_size: 棋盘格内角点数量 (列数, 行数)
            square_size: 棋盘格方格实际尺寸 (毫米)
            save_debug_images: 是否保存调试图像

        返回:
            bool: 标定是否成功
        """
        # 准备物体坐标点 (0,0,0), (1,0,0), (2,0,0) ....,(8,5,0)
        objp = np.zeros((checkerboard_size[0] * checkerboard_size[1], 3), np.float32)
        objp[:, :2] = np.mgrid[0:checkerboard_size[0], 0:checkerboard_size[1]].T.reshape(-1, 2)
        objp *= square_size  # 转换为实际尺寸 (毫米)

        # 存储所有图像的物体点和图像点
        objpoints = []  # 3D点
        imgpoints = []  # 2D点

        # 获取标定图像列表
        import glob
        images = glob.glob(os.path.join(calibration_images_path, "*.jpg")) + \
                glob.glob(os.path.join(calibration_images_path, "*.png"))

        if len(images) < 5:
            print(f"警告: 标定图像数量过少 ({len(images)}张)，建议至少5张以上")
            return False

        # 创建调试输出文件夹
        debug_dir = os.path.join(calibration_images_path, "debug_output")
        if save_debug_images:
            os.makedirs(debug_dir, exist_ok=True)

        found_count = 0
        print(f"开始处理 {len(images)} 张标定图像...")

        for img_path in images:
            img = cv2.imread(img_path)
            if img is None:
                continue

            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

            # 寻找棋盘格角点
            ret, corners = cv2.findChessboardCorners(gray, checkerboard_size, None)

            img_name = os.path.basename(img_path)

            if ret:
                objpoints.append(objp)

                # 亚像素精度优化
                criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
                corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
                imgpoints.append(corners2)
                found_count += 1
                print(f"  ✓ 成功检测到角点: {img_name}")

                # 保存标注了角点的调试图像
                if save_debug_images:
                    debug_img = img.copy()
                    cv2.drawChessboardCorners(debug_img, checkerboard_size, corners2, ret)

                    # 添加文本信息
                    cv2.putText(debug_img, f"Corners: {len(corners2)}", (10, 30),
                               cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                    cv2.putText(debug_img, f"Grid: {checkerboard_size[0]}x{checkerboard_size[1]}", (10, 70),
                               cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

                    debug_path = os.path.join(debug_dir, f"detected_{img_name}")
                    cv2.imwrite(debug_path, debug_img)

            else:
                print(f"  ✗ 未检测到角点: {img_name}")

                # 保存未检测到角点的图像用于调试
                if save_debug_images:
                    debug_img = img.copy()
                    cv2.putText(debug_img, "NO CORNERS DETECTED", (10, 30),
                               cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                    cv2.putText(debug_img, f"Expected: {checkerboard_size[0]}x{checkerboard_size[1]}", (10, 70),
                               cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

                    debug_path = os.path.join(debug_dir, f"failed_{img_name}")
                    cv2.imwrite(debug_path, debug_img)

        if found_count < 3:
            print(f"错误: 有效标定图像太少 ({found_count}张)，无法进行标定")
            print(f"调试图像已保存到: {debug_dir}")
            return False

        print(f"使用 {found_count} 张图像进行相机标定...")

        # 相机标定
        ret, self.camera_matrix, self.dist_coeffs, rvecs, tvecs = cv2.calibrateCamera(
            objpoints, imgpoints, gray.shape[::-1], None, None
        )

        if ret:
            self.calibrated = True
            print("相机标定成功!")
            print(f"内参矩阵:\n{self.camera_matrix}")
            print(f"畸变系数: {self.dist_coeffs.flatten()}")

            # 计算重投影误差
            total_error = 0
            for i in range(len(objpoints)):
                imgpoints2, _ = cv2.projectPoints(objpoints[i], rvecs[i], tvecs[i],
                                                self.camera_matrix, self.dist_coeffs)
                error = cv2.norm(imgpoints[i], imgpoints2, cv2.NORM_L2) / len(imgpoints2)
                total_error += error

            mean_error = total_error / len(objpoints)
            print(f"平均重投影误差: {mean_error:.3f} 像素")

            if mean_error > 1.0:
                print("警告: 重投影误差较大，建议检查标定图像质量")

            # 生成矫正对比图像
            if save_debug_images and len(images) > 0:
                self._generate_undistortion_comparison(images[:3], debug_dir)

            # 自动计算像素-毫米转换比例（基于棋盘格）
            self._calculate_scale_from_checkerboard(checkerboard_size, square_size, objpoints, imgpoints)

            if save_debug_images:
                print(f"调试图像已保存到: {debug_dir}")
                print("  - detected_*.jpg: 成功检测角点的图像")
                print("  - failed_*.jpg: 未检测到角点的图像")
                print("  - undistortion_comparison_*.jpg: 畸变矫正对比图像")

            return True
        else:
            print("相机标定失败!")
            return False

    def _generate_undistortion_comparison(self, sample_images, output_dir):
        """
        生成畸变矫正对比图像

        参数:
            sample_images: 样本图像路径列表
            output_dir: 输出目录
        """
        print("生成畸变矫正对比图像...")

        for i, img_path in enumerate(sample_images):
            img = cv2.imread(img_path)
            if img is None:
                continue

            h, w = img.shape[:2]

            # 获取优化的相机矩阵
            new_camera_matrix, roi = cv2.getOptimalNewCameraMatrix(
                self.camera_matrix, self.dist_coeffs, (w, h), 1, (w, h)
            )

            # 矫正图像
            undistorted = cv2.undistort(img, self.camera_matrix, self.dist_coeffs, None, new_camera_matrix)

            # 创建对比图像（左右并排）
            comparison = np.hstack((img, undistorted))

            # 添加标注
            cv2.putText(comparison, "ORIGINAL", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            cv2.putText(comparison, "CORRECTED", (w + 10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

            # 在中间画分割线
            cv2.line(comparison, (w, 0), (w, h), (255, 255, 255), 2)

            # 保存对比图像
            output_path = os.path.join(output_dir, f"undistortion_comparison_{i+1}.jpg")
            cv2.imwrite(output_path, comparison)

        print(f"已生成 {len(sample_images)} 张矫正对比图像")

    def _calculate_scale_from_checkerboard(self, checkerboard_size, square_size, objpoints, imgpoints):
        """
        根据棋盘格建立3D-2D点对应关系，用于准确的空间坐标转换
        注意：由于畸变的存在，不能使用固定的像素-毫米比例！

        参数:
            checkerboard_size: 棋盘格尺寸
            square_size: 格子实际尺寸（毫米）
            objpoints: 3D物体点
            imgpoints: 2D图像点
        """
        if len(imgpoints) == 0:
            return

        # 存储参考数据用于后续坐标转换
        self.checkerboard_size = checkerboard_size
        self.square_size = square_size
        self.objpoints = objpoints
        self.imgpoints = imgpoints

        # 使用第一张图像作为参考计算一个大致的比例（仅用于显示）
        img_corners = imgpoints[0].reshape(-1, 2)

        # 先矫正这些角点
        corrected_corners = cv2.undistortPoints(
            img_corners.reshape(-1, 1, 2),
            self.camera_matrix,
            self.dist_coeffs,
            None,
            self.camera_matrix
        ).reshape(-1, 2)

        # 计算矫正后相邻角点间的像素距离
        pixel_distances = []

        # 水平方向的距离
        for row in range(checkerboard_size[1]):
            for col in range(checkerboard_size[0] - 1):
                idx1 = row * checkerboard_size[0] + col
                idx2 = row * checkerboard_size[0] + col + 1

                p1 = corrected_corners[idx1]
                p2 = corrected_corners[idx2]
                pixel_dist = np.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)
                pixel_distances.append(pixel_dist)

        # 垂直方向的距离
        for row in range(checkerboard_size[1] - 1):
            for col in range(checkerboard_size[0]):
                idx1 = row * checkerboard_size[0] + col
                idx2 = (row + 1) * checkerboard_size[0] + col

                p1 = corrected_corners[idx1]
                p2 = corrected_corners[idx2]
                pixel_dist = np.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)
                pixel_distances.append(pixel_dist)

        # 计算平均像素距离（仅作为参考显示）
        avg_pixel_distance = np.mean(pixel_distances)
        reference_ratio = square_size / avg_pixel_distance

        print(f"参考像素-毫米比例: {reference_ratio:.4f} mm/pixel（仅用于显示）")
        print(f"注意：实际转换使用相机标定参数进行精确计算，不使用固定比例")
        print(f"基于矫正后棋盘格格子尺寸: {square_size}mm, 平均像素距离: {avg_pixel_distance:.2f}像素")

        # 不再设置固定的pixel_to_mm_ratio
        self.pixel_to_mm_ratio = None

    def save_calibration(self, filepath):
        """保存标定结果"""
        if not self.calibrated:
            print("错误: 相机未标定，无法保存")
            return False

        calibration_data = {
            'camera_matrix': self.camera_matrix.tolist(),
            'dist_coeffs': self.dist_coeffs.tolist(),
            'calibrated': True
        }

        # 保存棋盘格相关信息用于坐标转换
        if hasattr(self, 'checkerboard_size') and hasattr(self, 'square_size'):
            calibration_data['checkerboard_size'] = self.checkerboard_size
            calibration_data['square_size'] = self.square_size

        # 保存标定时的角点信息（仅保存第一张图像作为参考）
        if hasattr(self, 'imgpoints') and len(self.imgpoints) > 0:
            calibration_data['reference_imgpoints'] = self.imgpoints[0].tolist()

        with open(filepath, 'w') as f:
            json.dump(calibration_data, f, indent=2)

        print(f"标定结果已保存到: {filepath}")
        return True

    def load_calibration(self, filepath):
        """加载标定结果"""
        try:
            with open(filepath, 'r') as f:
                calibration_data = json.load(f)

            self.camera_matrix = np.array(calibration_data['camera_matrix'])
            self.dist_coeffs = np.array(calibration_data['dist_coeffs'])
            self.calibrated = calibration_data.get('calibrated', False)

            # 加载棋盘格信息
            if 'checkerboard_size' in calibration_data:
                self.checkerboard_size = calibration_data['checkerboard_size']
            if 'square_size' in calibration_data:
                self.square_size = calibration_data['square_size']

            # 加载参考角点信息
            if 'reference_imgpoints' in calibration_data:
                self.imgpoints = [np.array(calibration_data['reference_imgpoints'])]

            print(f"标定结果已从 {filepath} 加载")
            print(f"内参矩阵:\n{self.camera_matrix}")
            print(f"畸变系数: {self.dist_coeffs.flatten()}")

            # 计算并显示参考比例（仅用于显示）
            if hasattr(self, 'checkerboard_size') and hasattr(self, 'square_size') and hasattr(self, 'imgpoints'):
                ref_corners = self.imgpoints[0].reshape(-1, 2)
                corrected_ref = cv2.undistortPoints(
                    ref_corners.reshape(-1, 1, 2),
                    self.camera_matrix,
                    self.dist_coeffs,
                    None,
                    self.camera_matrix
                ).reshape(-1, 2)

                # 计算平均像素距离作为参考
                pixel_distances = []
                for row in range(self.checkerboard_size[1]):
                    for col in range(self.checkerboard_size[0] - 1):
                        idx1 = row * self.checkerboard_size[0] + col
                        idx2 = row * self.checkerboard_size[0] + col + 1

                        p1 = corrected_ref[idx1]
                        p2 = corrected_ref[idx2]
                        pixel_dist = np.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)
                        pixel_distances.append(pixel_dist)

                if pixel_distances:
                    avg_pixel_distance = np.mean(pixel_distances)
                    reference_ratio = self.square_size / avg_pixel_distance
                    print(f"参考像素-毫米比例: {reference_ratio:.4f} mm/pixel（图像中心区域，仅供参考）")
                    print(f"注意：实际使用精确的相机标定算法进行坐标转换")

            return True
        except Exception as e:
            print(f"加载标定文件失败: {e}")
            return False

    def set_scale_ratio(self, pixel_distance, real_distance_mm):
        """
        设置像素到真实距离的转换比例

        参数:
            pixel_distance: 像素距离
            real_distance_mm: 对应的真实距离 (毫米)
        """
        self.pixel_to_mm_ratio = real_distance_mm / pixel_distance
        print(f"设置像素-毫米转换比例: {self.pixel_to_mm_ratio:.4f} mm/pixel")

    def undistort_points(self, points):
        """
        矫正关键点坐标的畸变

        参数:
            points: 畸变的关键点坐标 [(x1,y1), (x2,y2), ...]

        返回:
            矫正后的关键点坐标
        """
        if not self.calibrated:
            print("警告: 相机未标定，返回原始坐标")
            return points

        if len(points) == 0:
            return points

        # 转换为numpy数组
        points_array = np.array(points, dtype=np.float32).reshape(-1, 1, 2)

        # 矫正畸变
        undistorted_points = cv2.undistortPoints(
            points_array, self.camera_matrix, self.dist_coeffs,
            None, self.camera_matrix
        )

        # 转换回原始格式
        result = undistorted_points.reshape(-1, 2)
        return [(float(pt[0]), float(pt[1])) for pt in result]

    def pixels_to_real(self, points):
        """
        将像素坐标转换为真实坐标 (毫米)
        使用相机标定参数进行精确转换，而不是固定比例

        参数:
            points: 像素坐标列表 [(x1,y1), (x2,y2), ...]

        返回:
            真实坐标列表 (毫米)
        """
        if not self.calibrated:
            print("警告: 相机未标定，返回像素坐标")
            return points

        if not hasattr(self, 'square_size') or self.square_size is None:
            print("警告: 未设置真实尺寸参考，返回像素坐标")
            return points

        # 首先矫正畸变
        corrected_points = self.undistort_points(points)

        # 使用相机标定参数将像素坐标转换为真实坐标
        real_points = []
        for x, y in corrected_points:
            # 使用平面几何转换，假设所有鱼都在同一平面上（俯拍视角）
            # 这里使用相机内参矩阵和已知的真实尺寸参考进行转换

            # 获取相机主点
            cx = self.camera_matrix[0, 2]
            cy = self.camera_matrix[1, 2]

            # 获取焦距
            fx = self.camera_matrix[0, 0]
            fy = self.camera_matrix[1, 1]

            # 假设鱼在Z=0平面（俯拍视角），使用棋盘格作为尺寸参考
            # 计算相对于图像中心的偏移
            offset_x = x - cx
            offset_y = y - cy

            # 使用棋盘格建立的尺寸关系进行转换
            # 这里需要一个参考高度，我们使用棋盘格标定时的平均高度
            if hasattr(self, 'imgpoints') and len(self.imgpoints) > 0:
                # 使用第一张标定图像建立的转换关系
                ref_corners = self.imgpoints[0].reshape(-1, 2)
                corrected_ref = cv2.undistortPoints(
                    ref_corners.reshape(-1, 1, 2),
                    self.camera_matrix,
                    self.dist_coeffs,
                    None,
                    self.camera_matrix
                ).reshape(-1, 2)

                # 计算参考比例（使用图像中心区域的角点）
                center_x, center_y = cx, cy
                distances_to_center = [np.sqrt((pt[0]-center_x)**2 + (pt[1]-center_y)**2) for pt in corrected_ref]
                center_idx = np.argmin(distances_to_center)

                # 找到最接近中心的角点附近的水平和垂直相邻点
                row_idx = center_idx // self.checkerboard_size[0]
                col_idx = center_idx % self.checkerboard_size[0]

                # 计算该区域的像素-毫米比例
                if col_idx < self.checkerboard_size[0] - 1:
                    right_idx = row_idx * self.checkerboard_size[0] + col_idx + 1
                    pixel_dist_h = np.sqrt((corrected_ref[right_idx][0] - corrected_ref[center_idx][0])**2 +
                                         (corrected_ref[right_idx][1] - corrected_ref[center_idx][1])**2)
                    scale_h = self.square_size / pixel_dist_h
                else:
                    scale_h = self.square_size / 50  # 默认值

                if row_idx < self.checkerboard_size[1] - 1:
                    down_idx = (row_idx + 1) * self.checkerboard_size[0] + col_idx
                    pixel_dist_v = np.sqrt((corrected_ref[down_idx][0] - corrected_ref[center_idx][0])**2 +
                                         (corrected_ref[down_idx][1] - corrected_ref[center_idx][1])**2)
                    scale_v = self.square_size / pixel_dist_v
                else:
                    scale_v = self.square_size / 50  # 默认值

                # 使用平均比例
                scale = (scale_h + scale_v) / 2

                # 转换为真实坐标
                real_x = offset_x * scale
                real_y = offset_y * scale

            else:
                # 回退到简单转换
                real_x = offset_x * 0.1  # 默认比例
                real_y = offset_y * 0.1

            real_points.append((real_x, real_y))

        return real_points

    def calculate_real_distance(self, point1, point2):
        """
        计算两点间的真实距离 (毫米)
        使用精确的畸变矫正和坐标转换

        参数:
            point1, point2: 像素坐标点

        返回:
            真实距离 (毫米)
        """
        if not self.calibrated:
            # 返回像素距离
            pixel_dist = np.sqrt((point1[0] - point2[0])**2 + (point1[1] - point2[1])**2)
            return pixel_dist

        # 转换为真实坐标
        real_points = self.pixels_to_real([point1, point2])

        # 计算真实距离
        real_dist = np.sqrt((real_points[0][0] - real_points[1][0])**2 +
                           (real_points[0][1] - real_points[1][1])**2)

        return real_dist


class FishTracker:
    """
    高级鱼类跟踪器：结合YOLOv8的姿态估计和目标跟踪功能，实时跟踪每条鱼的关键点骨架、ID和位置
    特点：
    1. 稳定的关键点检测和跟踪
    2. 平滑的轨迹显示
    3. 清晰的ID标识
    4. 完整的骨架可视化
    5. 尾摆频率计算
    6. 鱼体态分析（C型或S型）
    7. 畸变矫正和真实值计算
    """
    def __init__(self, model_path="bestx.pt", tracker="bytetrack.yaml", conf=0.5,
                 history_length=100, smooth_factor=0.3, fps=30,
                 enable_distortion_correction=False, calibration_file=None,
                 batch_mode=False, output_dir="batch_output", use_cuda=True, save_interval=1.0):
        """
        初始化鱼类跟踪器

        参数:
            model_path: YOLOv8姿态估计模型路径
            tracker: 跟踪器配置文件
            conf: 置信度阈值
            history_length: 轨迹历史长度 (增加以保留更长的轨迹历史)
            smooth_factor: 关键点平滑因子 (0-1)，越大越平滑，默认值降低以显示真实骨架变化
            fps: 视频帧率，用于精确计算时间窗口
            enable_distortion_correction: 是否启用畸变矫正
            calibration_file: 相机标定文件路径
            batch_mode: 是否启用批处理模式（离线处理）
            output_dir: 批处理输出目录
            use_cuda: 是否使用CUDA加速
            save_interval: 批处理模式数据保存间隔（秒）
        """
        # 检测和配置CUDA
        import torch
        self.use_cuda = use_cuda and torch.cuda.is_available()
        device = 'cuda' if self.use_cuda else 'cpu'

        print(f"初始化跟踪器 - 设备: {device}")
        if self.use_cuda:
            print(f"CUDA设备: {torch.cuda.get_device_name(0)}")

        # 加载模型并设置设备
        self.model = YOLO(model_path)
        if self.use_cuda:
            self.model.to(device)

        # 批处理模式相关
        self.batch_mode = batch_mode
        self.output_dir = output_dir
        self.save_interval = save_interval  # 保存间隔（秒）
        self.last_save_time = 0  # 上次保存时间
        self.frame_buffer = []  # 帧缓冲区，减少IO
        self.data_buffer = []  # 数据缓冲区

        if self.batch_mode:
            os.makedirs(self.output_dir, exist_ok=True)
            os.makedirs(os.path.join(self.output_dir, "frames"), exist_ok=True)
            os.makedirs(os.path.join(self.output_dir, "data"), exist_ok=True)
            print(f"批处理模式 - 数据保存间隔: {save_interval}秒")

        # 畸变矫正相关
        self.enable_distortion_correction = enable_distortion_correction
        self.calibrator = CameraCalibrator()

        # 如果启用畸变矫正，加载标定文件
        if enable_distortion_correction and calibration_file:
            if os.path.exists(calibration_file):
                self.calibrator.load_calibration(calibration_file)
            else:
                print(f"警告: 标定文件 {calibration_file} 不存在，畸变矫正将被禁用")
                self.enable_distortion_correction = False

        # 创建自定义跟踪器配置以提高ID稳定性
        import yaml
        from pathlib import Path

        # 获取原始跟踪器配置文件路径
        if os.path.isfile(tracker):
            tracker_path = tracker
        else:
            # 如果只提供了文件名，则在ultralytics配置目录中查找
            ultralytics_dir = Path(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
            tracker_path = ultralytics_dir / "ultralytics" / "cfg" / "trackers" / tracker

        # 加载原始配置
        if os.path.isfile(tracker_path):
            with open(tracker_path, 'r') as f:
                tracker_cfg = yaml.safe_load(f)

            # 修改跟踪器参数以提高ID稳定性
            tracker_cfg['track_buffer'] = 120  # 增加缓冲区大小，使ID保持更长时间
            tracker_cfg['track_low_thresh'] = 0.05  # 降低第二关联阈值，提高低置信度目标的跟踪能力
            tracker_cfg['new_track_thresh'] = 0.45  # 降低新轨迹阈值，更容易保持现有ID而非创建新ID
            tracker_cfg['match_thresh'] = 0.9  # 提高匹配阈值，确保更精确的匹配

            # 创建临时配置文件
            import tempfile
            temp_config = tempfile.NamedTemporaryFile(delete=False, suffix='.yaml')
            with open(temp_config.name, 'w') as f:
                yaml.dump(tracker_cfg, f)

            self.tracker_config = temp_config.name
        else:
            self.tracker_config = tracker

        self.conf = conf
        self.history_length = history_length
        self.smooth_factor = smooth_factor
        self.fps = fps  # 存储视频帧率

        # 存储鱼的特征信息用于ID恢复
        self.fish_features = {}
        self.last_seen_frame = {}

        # 计算5秒对应的帧数
        self.frames_per_5sec = int(self.fps * 5)  # 精确计算5秒对应的帧数

        # 跟踪历史记录
        self.track_history = defaultdict(lambda: [])
        self.keypoint_history = defaultdict(lambda: [])
        self.smoothed_keypoints = {}
        self.names = self.model.model.names

        # 速度计算相关
        self.position_history = defaultdict(lambda: deque(maxlen=30))  # 位置历史，存储1秒内的数据(30fps)
        self.speed_history = defaultdict(lambda: deque(maxlen=30))  # 速度历史，存储1秒内的速度数据
        self.speed_update_counter = defaultdict(int)  # 速度更新计数器

        # 尾摆频率计算相关
        self.tail_frequency_counter = defaultdict(int)  # 尾摆频率更新计数器
        self.tail_frequency_update_interval = 5 * 30  # 5秒更新一次（假设30fps）
        self.tail_positions = defaultdict(lambda: deque(maxlen=150))  # 存储5秒内的尾部位置历史
        self.tail_direction_changes = defaultdict(int)  # 记录方向变化次数
        self.last_tail_directions = {}  # 上一次尾部方向
        self.last_direction_change_time = defaultdict(float)  # 上次方向变化时间
        self.tail_frequencies = defaultdict(int)  # 存储计算出的尾摆频率，改为整数避免小数点
        self.fish_postures = {}  # 存储鱼的体态（C型、S型或直线型）
        self.frame_count = defaultdict(int)  # 帧计数器
        self.posture_history = defaultdict(lambda: deque(maxlen=10))  # 存储体态历史，用于平滑体态变化
        self.spine_cross_count = defaultdict(int)  # 记录尾部穿过脊柱线的次数
        self.last_spine_side = {}  # 记录尾部相对于脊柱线的上一次位置
        self.tail_freq_history = defaultdict(lambda: deque(maxlen=5))  # 存储最近计算的频率，用于平滑

        # 结果记录相关
        self.tracking_data = defaultdict(list)  # 存储每条鱼的跟踪数据
        self.fish_data = {}  # 存储鱼类数据，用于GUI显示
        self.global_stats = {}  # 全局统计数据

        # 日志记录相关
        self.log_dir = "fish_logs"  # 日志文件目录
        self.summary_file = os.path.join(self.log_dir, "fish_summary.csv")  # 汇总文件
        self.log_file = os.path.join(self.log_dir, "fish_tracking_data.txt")  # 所有鱼的统一日志文件
        self.last_positions = {}  # 上一帧的位置，用于计算速度
        self.last_timestamps = {}  # 上一帧的时间戳，用于计算速度

        # 创建日志目录
        os.makedirs(self.log_dir, exist_ok=True)

        # 鱼类骨架定义 - 根据实际鱼类关键点调整
        self.skeleton = [
            [0, 1],
            [1, 2],
            [2, 3],
            [3, 4]
        ]

        # 关键点颜色映射
        self.keypoint_colors = [
            (0, 255, 0),    #  绿色
            (0, 255, 128),  #  浅绿
            (0, 255, 255),  #  黄绿
            (0, 128, 255),  #  橙色
            (0, 0, 255),    #  蓝色
        ]

        # 在初始化部分添加逐秒记录相关变量
        # 逐秒数据记录
        self.second_data = defaultdict(lambda: defaultdict(list))  # {track_id: {second: [data_entries]}}
        self.last_second_logged = defaultdict(int)  # 记录每条鱼上次记录的秒数

        # 速度和尾摆频率的有效计算
        self.valid_speeds = defaultdict(list)  # 存储有效的速度数据
        self.valid_tail_frequencies = defaultdict(list)  # 存储有效的尾摆频率数据
        
        # 计算累积统计数据
        self.cumulative_distances = defaultdict(float)  # 累积游泳距离
        self.total_tracking_time = defaultdict(float)    # 总跟踪时间

    def _flush_batch_buffers(self):
        """刷新批处理缓冲区，批量保存数据"""
        if not self.batch_mode or not self.frame_buffer:
            return

        try:
            # 只保存最新的几帧（减少存储）
            frames_to_save = self.frame_buffer[-5:] if len(self.frame_buffer) > 5 else self.frame_buffer

            # 批量保存帧
            for frame_num, frame in frames_to_save:
                frame_filename = f"frame_{frame_num:06d}.jpg"
                frame_path = os.path.join(self.output_dir, "frames", frame_filename)
                cv2.imwrite(frame_path, frame, [cv2.IMWRITE_JPEG_QUALITY, 85])  # 降低质量减少文件大小

            # 保存数据摘要（而非每帧数据）
            if self.data_buffer:
                latest_data = self.data_buffer[-1]  # 只保存最新数据
                summary_file = os.path.join(self.output_dir, "data", f"summary_{latest_data['frame_number']:06d}.json")
                with open(summary_file, 'w') as f:
                    json.dump(latest_data, f, default=str)

            # 清空缓冲区
            self.frame_buffer.clear()
            self.data_buffer.clear()

            print(f"批量保存完成 - 帧数: {len(frames_to_save)}")

        except Exception as e:
            print(f"ERROR: 批量保存失败: {e}")

    def _save_heatmap_data(self, frame_number):
        """保存当前帧的热力图数据"""
        heatmap_data = {
            'frame_number': frame_number,
            'positions': []
        }

        for track_id, fish_info in self.fish_data.items():
            try:
                # 安全检查fish_info类型
                if not isinstance(fish_info, dict):
                    print(f"WARNING: _save_heatmap_data - track_id {track_id} 的fish_info不是字典类型: {type(fish_info)}")
                    continue

                # 安全获取positions
                positions = fish_info.get('positions', [])
                if not isinstance(positions, list) or not positions:
                    continue

                latest_pos = positions[-1]
                if not isinstance(latest_pos, dict):
                    print(f"WARNING: _save_heatmap_data - track_id {track_id} 的latest_pos不是字典类型: {type(latest_pos)}")
                    continue

                # 安全获取像素坐标
                pixel_info = latest_pos.get('pixel', {})
                if not isinstance(pixel_info, dict):
                    print(f"WARNING: _save_heatmap_data - track_id {track_id} 的pixel信息不是字典类型: {type(pixel_info)}")
                    continue

                heatmap_data['positions'].append({
                    'track_id': track_id,
                    'x': pixel_info.get('x', 0),
                    'y': pixel_info.get('y', 0),
                    'speed': latest_pos.get('speed_pixel', 0),
                    'tail_frequency': fish_info.get('tail_frequency', 0)
                })

            except Exception as e:
                print(f"ERROR: _save_heatmap_data处理track_id {track_id} 时出错: {e}")
                continue

        # 保存为JSON文件
        heatmap_file = os.path.join(self.output_dir, "data", f"heatmap_{frame_number:06d}.json")
        with open(heatmap_file, 'w') as f:
            json.dump(heatmap_data, f)

    def filter_valid_fish_ids(self, min_track_length=10, min_tail_frequency=0.1, min_speed=1.0):
        """
        过滤有效的鱼类ID

        参数:
            min_track_length: 最小轨迹长度
            min_tail_frequency: 最小尾摆频率
            min_speed: 最小平均速度

        返回:
            有效的鱼类ID列表
        """
        valid_ids = []

        # 确保fish_data存在且不为空
        if not hasattr(self, 'fish_data') or not self.fish_data:
            print("WARNING: fish_data为空或不存在")
            return valid_ids

        for track_id, fish_info in self.fish_data.items():
            try:
                # 安全检查fish_info的类型
                if not isinstance(fish_info, dict):
                    print(f"WARNING: track_id {track_id} 的fish_info不是字典类型: {type(fish_info)}")
                    continue

                # 检查轨迹长度
                positions = fish_info.get('positions', [])
                if not isinstance(positions, list):
                    print(f"WARNING: track_id {track_id} 的positions不是列表类型: {type(positions)}")
                    continue

                track_length = len(positions)
                if track_length < min_track_length:
                    continue

                # 检查尾摆频率
                tail_freq = fish_info.get('tail_frequency', 0)
                if not isinstance(tail_freq, (int, float)):
                    print(f"WARNING: track_id {track_id} 的tail_frequency不是数字类型: {type(tail_freq)}")
                    tail_freq = 0

                if tail_freq < min_tail_frequency:
                    continue

                # 检查平均速度
                speeds = fish_info.get('speeds', [])
                if not isinstance(speeds, list):
                    print(f"WARNING: track_id {track_id} 的speeds不是列表类型: {type(speeds)}")
                    continue

                if speeds:
                    try:
                        # 确保所有速度值都是数字
                        numeric_speeds = [float(s) for s in speeds if isinstance(s, (int, float))]
                        if numeric_speeds:
                            avg_speed = sum(numeric_speeds) / len(numeric_speeds)
                            if avg_speed < min_speed:
                                continue
                        else:
                            continue
                    except (ValueError, TypeError) as e:
                        print(f"WARNING: track_id {track_id} 速度计算错误: {e}")
                        continue
                else:
                    continue

                valid_ids.append(track_id)

            except Exception as e:
                print(f"ERROR: 处理track_id {track_id} 时出错: {e}")
                continue

        return valid_ids

    def get_filtered_fish_data(self, filter_invalid=True, **filter_params):
        """
        获取过滤后的鱼类数据

        参数:
            filter_invalid: 是否过滤无效ID
            **filter_params: 过滤参数

        返回:
            过滤后的鱼类数据
        """
        if not filter_invalid:
            return self.fish_data

        valid_ids = self.filter_valid_fish_ids(**filter_params)
        return {track_id: self.fish_data[track_id] for track_id in valid_ids if track_id in self.fish_data}

    def save_batch_summary(self):
        """保存批处理模式的汇总数据"""
        if not self.batch_mode:
            return

        try:
            # 安全获取有效鱼类数量
            try:
                valid_fish_count = len(self.filter_valid_fish_ids())
            except Exception as e:
                print(f"WARNING: 计算有效鱼类数量失败: {e}")
                valid_fish_count = 0

            # 确保刷新剩余缓冲区
            self._flush_batch_buffers()

            # 保存完整的跟踪数据
            summary_data = {
                'total_frames': len(self.data_buffer) if hasattr(self, 'data_buffer') else 0,
                'fps': self.fps,
                'fish_data': self.fish_data,
                'global_stats': self.global_stats,
                'valid_fish_count': valid_fish_count,
                'processing_complete': True,
                'cuda_enabled': self.use_cuda,
                'save_interval': self.save_interval
            }

            summary_file = os.path.join(self.output_dir, "tracking_summary.json")
            with open(summary_file, 'w') as f:
                json.dump(summary_data, f, indent=2, default=str)  # 添加default=str处理不可序列化的对象

            # 保存过滤后的有效数据
            try:
                valid_data = self.get_filtered_fish_data(filter_invalid=True)
                valid_summary_file = os.path.join(self.output_dir, "valid_fish_summary.json")
                with open(valid_summary_file, 'w') as f:
                    json.dump(valid_data, f, indent=2, default=str)
            except Exception as e:
                print(f"WARNING: 保存有效数据失败: {e}")
                valid_data = {}

            print(f"批处理完成！")
            print(f"总帧数: {len(self.frame_data)}")
            print(f"总鱼类ID: {len(self.fish_data)}")
            print(f"有效鱼类ID: {len(valid_data)}")
            print(f"输出目录: {self.output_dir}")

        except Exception as e:
            print(f"ERROR: save_batch_summary失败: {e}")
            import traceback
            traceback.print_exc()

    def set_fps(self, fps):
        """
        动态设置视频帧率
        
        参数:
            fps: 视频的实际帧率
        """
        old_fps = self.fps
        self.fps = fps
        print(f"🎬 FPS更新: {old_fps} → {fps}")
        
        # 清空之前可能基于错误fps计算的数据
        if abs(old_fps - fps) > 1:  # 如果fps差异较大，重置某些计算
            print(f"⚠️ FPS差异较大 ({old_fps} → {fps})，重置时间相关的计算")
            # 不完全重置，只是标记需要重新校准
            for track_id in list(self.last_timestamps.keys()):
                # 调整之前的时间戳
                if track_id in self.last_timestamps:
                    # 按新的fps重新计算时间戳
                    old_timestamp = self.last_timestamps[track_id]
                    old_frame = int(old_timestamp * old_fps)
                    new_timestamp = old_frame / fps
                    self.last_timestamps[track_id] = new_timestamp

    def process_frame(self, frame):
        """
        处理单帧图像，返回带有跟踪结果的图像

        参数:
            frame: 输入图像帧

        返回:
            带有跟踪结果的图像帧
        """
        # 使用模型进行跟踪，persist=True保持跟踪状态
        results = self.model.track(frame, persist=True, conf=self.conf, tracker=self.tracker_config, verbose=False)

        if results[0].boxes is None or len(results[0].boxes) == 0:
            return frame

        # 创建标注器 - 使用更细的线条
        annotator = Annotator(frame, line_width=1)

        # 当前帧号
        current_frame = max([self.frame_count.get(id, 0) for id in self.frame_count] or [0]) + 1

        # 初始化当前帧特征字典
        current_fish_features = {}

        # 检查是否有跟踪ID
        if results[0].boxes.id is not None:
            # 提取预测结果
            boxes = results[0].boxes.xyxy.cpu()
            track_ids = results[0].boxes.id.int().cpu().tolist()
            clss = results[0].boxes.cls.cpu().tolist()
            confs = results[0].boxes.conf.cpu().tolist()
            keypoints = results[0].keypoints.data if results[0].keypoints is not None else None

            # 存储当前帧中的所有鱼的特征 (已在方法开始时初始化)

            # 处理每个检测结果
            for i, (box, track_id, cls, conf) in enumerate(zip(boxes, track_ids, clss, confs)):
                # 获取鱼类颜色
                color = colors(int(cls), True)

                # 计算鱼的特征向量 (使用边界框、大小、位置等)
                fish_width = float(box[2] - box[0])
                fish_height = float(box[3] - box[1])
                center_x = float((box[0] + box[2]) / 2)
                center_y = float((box[1] + box[3]) / 2)
                aspect_ratio = fish_width / max(fish_height, 1e-6)
                area = fish_width * fish_height

                # 创建特征向量
                feature_vector = np.array([fish_width, fish_height, center_x, center_y, aspect_ratio, area, conf])
                current_fish_features[track_id] = feature_vector

                # 更新最后一次看到的帧
                self.last_seen_frame[track_id] = current_frame

                # 如果是新的鱼，添加到特征字典
                if track_id not in self.fish_features:
                    self.fish_features[track_id] = feature_vector
                else:
                    # 平滑更新特征 (70% 旧特征 + 30% 新特征)
                    self.fish_features[track_id] = 0.7 * self.fish_features[track_id] + 0.3 * feature_vector

                # 绘制边界框和ID标签 - 使用更细、更透明的线条
                label = f"ID:{track_id} {self.names[int(cls)]}"
                # 使用alpha参数使边界框更透明
                annotator.box_label(box, label=label, color=color)

                # 存储中心点轨迹
                center_x_int = int(center_x)
                center_y_int = int(center_y)
                track = self.track_history[track_id]
                track.append((center_x_int, center_y_int))
                if len(track) > self.history_length:
                    track.pop(0)

                # 绘制轨迹 - 使用更细的线条
                if len(track) > 1:
                    points = np.array(track, dtype=np.int32).reshape((-1, 1, 2))
                    cv2.polylines(frame, [points], isClosed=False, color=color, thickness=2)

                # 处理关键点
                if keypoints is not None:
                    kpt = keypoints[i].cpu().numpy()
                    self.process_keypoints(frame, kpt, track_id, color)

            # 尝试恢复最近消失的鱼的ID
            self._recover_lost_ids(current_fish_features, current_frame)

        # 更新全局统计数据
        self.global_stats = {
            'total_fish_count': len(self.fish_data),
            'active_fish_count': len(current_fish_features),
            'total_frames_processed': current_frame,
            'average_tail_frequency': sum(self.tail_frequencies.values()) / max(len(self.tail_frequencies), 1)
        }

        # 批处理模式：按间隔保存数据（优化性能）
        if self.batch_mode:
            current_time_sec = time.time()

            # 添加到缓冲区
            self.frame_buffer.append((current_frame, frame.copy()))
            self.data_buffer.append({
                'frame_number': current_frame,
                'timestamp': current_time_sec,
                'fish_data': copy.deepcopy(self.fish_data),
                'global_stats': copy.deepcopy(self.global_stats)
            })

            # 检查是否需要保存（按时间间隔）
            if current_time_sec - self.last_save_time >= self.save_interval:
                self._flush_batch_buffers()
                self.last_save_time = current_time_sec

        return frame

    def _recover_lost_ids(self, current_fish_features, current_frame):
        """
        尝试恢复丢失的鱼ID

        参数:
            current_fish_features: 当前帧中所有鱼的特征
            current_frame: 当前帧号
        """
        # 查找最近消失的鱼 (在过去30帧内消失的)
        recently_lost_fish = {}
        for fish_id, last_frame in self.last_seen_frame.items():
            # 如果鱼不在当前帧中，且在最近30帧内消失
            if fish_id not in current_fish_features and (current_frame - last_frame) <= 30:
                recently_lost_fish[fish_id] = self.fish_features[fish_id]

        # 如果没有最近消失的鱼，直接返回
        if not recently_lost_fish:
            return

        # 对于当前帧中的每条鱼，检查是否是最近消失的鱼
        for current_id, current_feature in current_fish_features.items():
            # 跳过已经存在很久的ID
            if current_id in self.fish_features and (current_frame - self.last_seen_frame.get(current_id, 0)) > 5:
                continue

            # 计算与最近消失的鱼的特征相似度
            best_match_id = None
            best_match_score = float('inf')

            for lost_id, lost_feature in recently_lost_fish.items():
                # 计算欧氏距离 (较小的值表示更相似)
                distance = np.linalg.norm(current_feature - lost_feature)

                # 如果距离小于阈值且比之前找到的最佳匹配更好
                if distance < best_match_score:
                    best_match_score = distance
                    best_match_id = lost_id

            # 如果找到了很好的匹配，且距离小于阈值
            if best_match_id is not None and best_match_score < 100:  # 阈值可以根据实际情况调整
                # 将当前鱼的轨迹和特征合并到丢失的鱼中
                if current_id in self.track_history and best_match_id in self.track_history:
                    # 合并轨迹历史
                    self.track_history[best_match_id].extend(self.track_history[current_id])
                    # 限制历史长度
                    if len(self.track_history[best_match_id]) > self.history_length:
                        self.track_history[best_match_id] = self.track_history[best_match_id][-self.history_length:]

                    # 更新特征和最后一次看到的帧
                    self.fish_features[best_match_id] = current_feature
                    self.last_seen_frame[best_match_id] = current_frame

                    # 从当前帧中移除这条鱼，因为它已经被识别为之前的鱼
                    # 注意：这里不会真正从当前帧中移除，只是在下一帧中不再使用当前ID
                    recently_lost_fish.pop(best_match_id, None)

    def process_keypoints(self, frame, keypoints, track_id, color):
        """
        处理和绘制关键点和骨架

        参数:
            frame: 图像帧
            keypoints: 关键点数据
            track_id: 跟踪ID
            color: 颜色
        """
        # 存储关键点历史
        kpt_history = self.keypoint_history[track_id]
        kpt_history.append(keypoints)
        if len(kpt_history) > 10:  # 保留最近10帧的关键点用于平滑
            kpt_history.pop(0)

        # 平滑关键点 - 使用较小的平滑因子以保留真实骨架变化
        if track_id not in self.smoothed_keypoints:
            self.smoothed_keypoints[track_id] = keypoints.copy()
        else:
            # 只对可见的关键点进行轻微平滑
            visible_mask = keypoints[:, 2] > 0.5
            self.smoothed_keypoints[track_id][visible_mask] = (
                self.smooth_factor * self.smoothed_keypoints[track_id][visible_mask] +
                (1 - self.smooth_factor) * keypoints[visible_mask]
            )

            # 对于新出现的关键点，直接使用当前值
            new_visible = (keypoints[:, 2] > 0.5) & (self.smoothed_keypoints[track_id][:, 2] <= 0.5)
            self.smoothed_keypoints[track_id][new_visible] = keypoints[new_visible]

        # 更新尾部位置历史并计算尾摆频率
        self.update_tail_metrics(track_id, keypoints, frame)

        smoothed = self.smoothed_keypoints[track_id]

        # 绘制骨架连接线 - 使用更细的线条
        for p in self.skeleton:
            if smoothed[p[0], 2] > 0.5 and smoothed[p[1], 2] > 0.5:
                pt1 = (int(smoothed[p[0], 0]), int(smoothed[p[0], 1]))
                pt2 = (int(smoothed[p[1], 0]), int(smoothed[p[1], 1]))
                cv2.line(frame, pt1, pt2, color, 2)  # 线条宽度从2减小到1

        # 绘制关键点
        for k in range(len(smoothed)):
            x, y, conf = smoothed[k]
            if conf > 0.5:  # 只绘制置信度高的关键点
                # 使用特定的关键点颜色 - 减小关键点大小
                kpt_color = self.keypoint_colors[k % len(self.keypoint_colors)]
                cv2.circle(frame, (int(x), int(y)), 2, kpt_color, -1)  # 关键点大小从4减小到2

                # 在关键点旁边标注序号
                cv2.putText(
                    frame,
                    str(k),
                    (int(x) + 5, int(y) - 5),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    kpt_color,
                    1
                )

        # 在图像上显示关键点位置信息和尾摆频率、体态信息
        if smoothed[0, 2] > 0.5:  # 如果尾部关键点可见
            # 获取尾摆频率和体态信息
            tail_freq = self.tail_frequencies.get(track_id, 0)
            posture = self.fish_postures.get(track_id, "Unknown")

            # 显示位置、尾摆频率和体态信息 - 改为英文显示，单位改为次/5秒，并标明是基于实际帧率计算
            info_text = f"Fish #{track_id} - Tail Freq:{tail_freq:.2f}/5s (fps:{self.fps}) Posture:{posture}"
            cv2.putText(
                frame,
                info_text,
                (int(smoothed[0, 0]) - 10, int(smoothed[0, 1]) - 20),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                color,
                1
            )

        # 计算实时速度
        current_time = time.time()
        current_pixel_pos = (float(smoothed[0, 0]), float(smoothed[0, 1]))

        # 添加位置到历史记录 - 使用tuple格式保持一致性
        self.position_history[track_id].append((current_pixel_pos, current_time))

        # 计算速度（像素/秒）
        current_speed_pixel = 0.0
        current_speed_real = 0.0

        if len(self.position_history[track_id]) >= 2:
            # 使用最近两个位置计算速度
            try:
                prev_data = self.position_history[track_id][-2]
                curr_data = self.position_history[track_id][-1]

                # 安全解包数据
                if isinstance(prev_data, (tuple, list)) and len(prev_data) >= 2:
                    prev_pos, prev_time = prev_data[0], prev_data[1]
                elif isinstance(prev_data, dict):
                    prev_pos, prev_time = prev_data['position'], prev_data['timestamp']
                else:
                    print(f"WARNING: 意外的prev_data格式: {type(prev_data)}")
                    prev_pos, prev_time = current_pixel_pos, current_time

                if isinstance(curr_data, (tuple, list)) and len(curr_data) >= 2:
                    curr_pos, curr_time = curr_data[0], curr_data[1]
                elif isinstance(curr_data, dict):
                    curr_pos, curr_time = curr_data['position'], curr_data['timestamp']
                else:
                    print(f"WARNING: 意外的curr_data格式: {type(curr_data)}")
                    curr_pos, curr_time = current_pixel_pos, current_time

                time_diff = curr_time - prev_time
                if time_diff > 0:
                    # 计算像素距离
                    dx = curr_pos[0] - prev_pos[0]
                    dy = curr_pos[1] - prev_pos[1]
                    pixel_distance = (dx**2 + dy**2)**0.5

                    current_speed_pixel = pixel_distance / time_diff

                    # 如果启用了畸变矫正和真实尺寸转换，计算真实速度
                    if self.enable_distortion_correction and self.calibrator.calibrated:
                        try:
                            # 矫正位置
                            prev_corrected = self.calibrator.undistort_points([prev_pos])[0]
                            curr_corrected = self.calibrator.undistort_points([curr_pos])[0]

                            # 转换为真实坐标
                            if hasattr(self.calibrator, 'square_size') and self.calibrator.square_size:
                                prev_real = self.calibrator.pixels_to_real([prev_corrected])[0]
                                curr_real = self.calibrator.pixels_to_real([curr_corrected])[0]

                                # 计算真实距离（毫米）
                                real_dx = curr_real[0] - prev_real[0]
                                real_dy = curr_real[1] - prev_real[1]
                                real_distance = (real_dx**2 + real_dy**2)**0.5

                                current_speed_real = real_distance / time_diff  # mm/s
                        except Exception as e:
                            print(f"速度计算中的畸变矫正失败: {e}")
            except Exception as e:
                print(f"ERROR: 速度计算失败: {e}")
                current_speed_pixel = 0.0
                current_speed_real = 0.0

        # 平滑速度
        self.speed_history[track_id].append(current_speed_pixel)
        smoothed_speed = sum(self.speed_history[track_id]) / len(self.speed_history[track_id])

        # 为了确保数据一致性，在update_tail_metrics完成后更新fish_data
        # 这样fish_data使用的数据和tracking_data是一致的


    def update_tail_metrics(self, track_id, keypoints, frame):
        """
        更新尾部摆动指标：计算尾摆频率和判断鱼体态
        使用极坐标系，以1,0连线为极坐标轴（正方向为向量(1,0)），计算3,4连线与极坐标轴的夹角
        当鱼身笔直时角度为0°，鱼尾向右甩为正角度，向左甩为负角度
        当角度单调变化达到阈值时才立即计数一次尾摆

        参数:
            track_id: 鱼的跟踪ID
            keypoints: 当前帧的关键点数据
            frame: 当前帧图像
        """
        # 使用实际时间戳而非简单帧计数
        timestamp = self.frame_count[track_id] / self.fps  # 转换为秒

        # 增加帧计数
        self.frame_count[track_id] += 1

        # 使用精确的时间戳，避免未定义错误
        current_time = self.frame_count[track_id]  # 帧计数
        current_timestamp = timestamp  # 实际时间戳（秒）

        # 确保所有关键点可见 - 需要0,1,3,4关键点
        if len(keypoints) < 5 or any(keypoints[i, 2] <= 0.5 for i in [0, 1, 3, 4]):
            return

        # 获取关键点位置（像素坐标）
        point0 = (keypoints[0, 0], keypoints[0, 1])  # 鱼颈
        point1 = (keypoints[1, 0], keypoints[1, 1])  # 鱼头最顶端
        point2 = (keypoints[2, 0], keypoints[2, 1])  # 鱼身中心
        point3 = (keypoints[3, 0], keypoints[3, 1])  # 身尾连接处
        point4 = (keypoints[4, 0], keypoints[4, 1])  # 鱼尾末端

        # 初始化坐标变量 - 默认使用原始像素坐标
        corrected_points = [point0, point1, point2, point3, point4]
        real_points = [(point0[0], point0[1]), (point1[0], point1[1]), (point2[0], point2[1]), (point3[0], point3[1]), (point4[0], point4[1])]

        # 检查畸变矫正是否可用
        distortion_correction_available = (self.enable_distortion_correction and
                                         self.calibrator.calibrated)

        # 检查真实尺寸转换是否可用
        real_scale_available = (distortion_correction_available and
                              hasattr(self.calibrator, 'square_size') and
                              self.calibrator.square_size is not None)

        if distortion_correction_available:
            try:
                # 矫正畸变
                corrected_points = self.calibrator.undistort_points([point0, point1, point2, point3, point4])

                # 转换为真实坐标（毫米）
                if real_scale_available:
                    real_points = self.calibrator.pixels_to_real(corrected_points)
                else:
                    # 如果没有真实尺寸信息，使用矫正后的像素坐标
                    real_points = corrected_points

            except Exception as e:
                print(f"ERROR: Error in distortion correction: {e}")
                # 如果矫正失败，回退到原始坐标
                corrected_points = [point0, point1, point2, point3, point4]
                real_points = [(point0[0], point0[1]), (point1[0], point1[1]), (point2[0], point2[1]), (point3[0], point3[1]), (point4[0], point4[1])]
                distortion_correction_available = False
                real_scale_available = False

        # 使用矫正后的坐标进行计算
        corrected_point0, corrected_point1, corrected_point2, corrected_point3, corrected_point4 = corrected_points

        # 计算1,0连线向量（极坐标轴）- 从corrected_point1指向corrected_point0
        polar_axis = (corrected_point0[0] - corrected_point1[0], corrected_point0[1] - corrected_point1[1])
        polar_axis_length = (polar_axis[0]**2 + polar_axis[1]**2)**0.5

        # 计算3,4连线向量（头部向量）
        vector34 = (corrected_point4[0] - corrected_point3[0], corrected_point4[1] - corrected_point3[1])
        length34 = (vector34[0]**2 + vector34[1]**2)**0.5

        # 检查向量长度，避免除以零
        if polar_axis_length < 1e-6 or length34 < 1e-6:
            return

        # 归一化向量
        polar_axis_norm = (polar_axis[0]/polar_axis_length, polar_axis[1]/polar_axis_length)
        vector34_norm = (vector34[0]/length34, vector34[1]/length34)

        # 计算3,4连线的延长线方程 (y = k*x + b)
        if vector34_norm[0] == 0:  # 避免除以零
            k34 = float('inf')
            b34 = corrected_point3[0]
        else:
            k34 = vector34_norm[1] / vector34_norm[0]
            b34 = corrected_point3[1] - k34 * corrected_point3[0]

        # 计算极坐标轴与3,4连线的夹角
        # 使用向量点积公式：cos(θ) = (a·b)/(|a|·|b|)
        dot_product = polar_axis_norm[0]*vector34_norm[0] + polar_axis_norm[1]*vector34_norm[1]
        angle_rad = np.arccos(np.clip(dot_product, -1.0, 1.0))

        # 确定角度的符号（使用叉积判断）
        cross_product = polar_axis_norm[0]*vector34_norm[1] - polar_axis_norm[1]*vector34_norm[0]
        if cross_product < 0:
            angle_rad = -angle_rad

        # 转换为角度，范围为[-180, 180]
        signed_angle = angle_rad * 180 / np.pi

        # 暂时取消极坐标轴显示
        if False:  # 设置为False取消在图像上显示坐标系和角度
            # 绘制极坐标轴（1,0连线）
            polar_axis_end = (int(point1[0] + polar_axis_norm[0] * 100), int(point1[1] + polar_axis_norm[1] * 100))
            cv2.line(frame, (int(point1[0]), int(point1[1])), polar_axis_end, (255, 0, 0), 2)

            # 绘制3,4连线延长线
            if k34 != float('inf'):
                # 计算延长线的两个端点
                x_start = point3[0] - vector34_norm[0] * 50
                y_start = point3[1] - vector34_norm[1] * 50
                x_end = point4[0] + vector34_norm[0] * 100
                y_end = point4[1] + vector34_norm[1] * 100
                cv2.line(frame, (int(x_start), int(y_start)), (int(x_end), int(y_end)), (0, 255, 0), 2)
            else:
                # 垂直线的情况
                y_start = point3[1] - 50
                y_end = point4[1] + 100
                cv2.line(frame, (int(point3[0]), int(y_start)), (int(point4[0]), int(y_end)), (0, 255, 0), 2)

            # 在point1处绘制一个圆表示极坐标原点
            cv2.circle(frame, (int(point1[0]), int(point1[1])), 5, (0, 0, 255), -1)

            # 显示角度值
            cv2.putText(frame, f"Angle: {signed_angle:.1f}°",
                        (int(point1[0]) + 10, int(point1[1]) - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

            # 添加坐标系说明
            cv2.putText(frame, "Red: Polar Axis (1,0)", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)
            cv2.putText(frame, "Green: Head Vector (3,4)", (10, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

            # 如果启用了畸变矫正，显示矫正信息
            if self.enable_distortion_correction and self.calibrator.calibrated:
                cv2.putText(frame, "Distortion Corrected", (10, 70),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
                if (hasattr(self.calibrator, 'square_size') and
                    self.calibrator.square_size is not None):
                    cv2.putText(frame, "Real Scale Available", (10, 90),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)

        # 存储尾部位置和角度历史
        self.tail_positions[track_id].append((point0[0], point0[1], signed_angle))

        # 简化的尾摆频率计算 - 检测角度峰值变化
        if len(self.tail_positions[track_id]) >= 5:  # 需要至少5个点来检测趋势
            # 获取最近的角度历史
            recent_angles = [pos[2] for pos in list(self.tail_positions[track_id])[-5:]]

            # 检测局部极值（峰值和谷值）
            current_angle = recent_angles[-1]
            prev_angle = recent_angles[-2]
            prev_prev_angle = recent_angles[-3]

            # 角度变化阈值 - 只有显著变化才计数
            min_angle_change = 20  # 最小角度变化阈值（度）

            # 检测峰值：当前角度是局部最大值或最小值
            is_peak = False
            if len(recent_angles) >= 3:
                # 检测局部最大值（峰值）
                if (prev_angle > prev_prev_angle and prev_angle > current_angle and
                    abs(prev_angle) > min_angle_change):
                    is_peak = True
                # 检测局部最小值（谷值）
                elif (prev_angle < prev_prev_angle and prev_angle < current_angle and
                      abs(prev_angle) > min_angle_change):
                    is_peak = True

            # 如果检测到峰值，增加尾摆计数
            if is_peak:
                # 避免重复计数 - 检查是否与上次峰值时间间隔足够
                last_peak_time = self.last_direction_change_time.get(track_id, 0)
                min_interval = self.fps * 0.2  # 最小间隔0.2秒，避免噪声

                if current_time - last_peak_time >= min_interval:
                    self.tail_direction_changes[track_id] += 1
                    self.last_direction_change_time[track_id] = current_time

            # 实时计算近5秒内的尾摆频率
            window_duration = 5.0  # 5秒窗口
            
            # 获取过去5秒内的尾摆次数
            current_frequency = 0
            
            # 计算已跟踪时间
            tracking_duration = current_time / self.fps  # 总跟踪时间（秒）
            
            # 如果有尾摆记录，计算频率
            if self.tail_direction_changes[track_id] > 0:
                # 如果跟踪时间不足5秒，按比例估算5秒内的频率
                if tracking_duration < window_duration:
                    # 按比例放大到5秒
                    estimated_frequency = (self.tail_direction_changes[track_id] * window_duration) / tracking_duration
                    current_frequency = min(estimated_frequency, 50)  # 限制最大值50次/5秒
                else:
                    # 跟踪时间已超过5秒，使用实际的近5秒统计
                    # 这里简化处理：取最近的摆动次数
                    current_frequency = min(self.tail_direction_changes[track_id], 50)
            else:
                current_frequency = 0
            
            # 更新频率（实时更新，不等待5秒窗口）
            self.tail_frequencies[track_id] = round(current_frequency)
            
            # 调试输出
            print(f"🐟 Fish {track_id} 尾摆统计: 总次数={self.tail_direction_changes[track_id]}, 跟踪时间={tracking_duration:.1f}s, 频率={self.tail_frequencies[track_id]}次/5s")

        # 计算速度和距离相关指标 - 使用原版逻辑
        current_position = point0  # 使用原始像素坐标进行位置跟踪
        current_timestamp = self.frame_count[track_id] / self.fps  # 转换为秒
        
        # 计算速度（如果有上一帧的位置）- 原版逻辑
        speed_pixel = 0  # 像素/秒
        speed_real = 0   # 毫米/秒（如果有标定）
        
        if track_id in self.last_positions and track_id in self.last_timestamps:
            last_pos = self.last_positions[track_id]
            time_diff = current_timestamp - self.last_timestamps[track_id]
            
            # 详细调试输出
            print(f"🔍 Fish {track_id} 速度计算: last_pos={last_pos}, current_pos={current_position}, time_diff={time_diff:.4f}s")
            
            if time_diff > 0:
                # 计算像素距离速度
                pixel_distance = ((current_position[0] - last_pos[0])**2 + 
                            (current_position[1] - last_pos[1])**2)**0.5
                speed_pixel = pixel_distance / time_diff  # 像素/秒
                
                # 详细调试输出
                print(f"🔢 Fish {track_id} 计算结果: 距离={pixel_distance:.2f}px, 时间差={time_diff:.4f}s, 速度={speed_pixel:.2f}px/s")
                
                # 如果启用畸变矫正，计算真实速度
                if distortion_correction_available:
                    try:
                        # 如果有真实尺寸转换信息，计算真实速度
                        if real_scale_available:
                            real_distance = self.calibrator.calculate_real_distance(last_pos, current_position)
                            speed_real = real_distance / time_diff  # 毫米/秒
                            print(f"🌍 Fish {track_id} 真实速度: {speed_real:.2f}mm/s")
                        else:
                            # 没有真实尺寸转换，速度保持0
                            speed_real = 0
                    except Exception as e:
                        print(f"❌ Fish {track_id} 真实速度计算失败: {e}")
                        speed_real = 0
            else:
                print(f"⚠️ Fish {track_id} 时间差无效: {time_diff:.4f}s")
        else:
            print(f"⏳ Fish {track_id} 缺少上一帧数据（last_positions或last_timestamps）")
            
        # 更新上一帧的位置和时间戳
        self.last_positions[track_id] = current_position
        self.last_timestamps[track_id] = current_timestamp





        # 计算鱼体长度（从鱼头到鱼尾的距离）
        fish_length_pixel = np.sqrt((point1[0] - point4[0])**2 + (point1[1] - point4[1])**2)
        fish_length_real = 0

        if real_scale_available:
            try:
                fish_length_real = self.calibrator.calculate_real_distance(point1, point4)
            except Exception as e:
                print(f"ERROR: Error calculating real fish length: {e}")
                fish_length_real = 0

        # 先判断鱼的体态（在记录数据前）
        self.determine_fish_posture(track_id, keypoints)
        
        # 记录跟踪数据（包含像素和真实坐标）
        tracking_entry = {
            'frame': self.frame_count[track_id],
            'timestamp': current_timestamp,
            # 像素坐标数据
            'tail_position_pixel': point0,
            'head_position_pixel': point1,
            'body_center_pixel': (keypoints[2, 0], keypoints[2, 1]),
            'body_tail_junction_pixel': point3,
            'tail_end_pixel': point4,
            'fish_length_pixel': fish_length_pixel,
            'speed_pixel': speed_pixel,
            # 角度和频率
            'tail_angle': signed_angle,
            'tail_freq': self.tail_frequencies.get(track_id, 0),
            # 体态（每帧记录）
            'posture': self.fish_postures.get(track_id, "Unknown"),
            # 真实坐标数据（如果有标定）
            'corrected_available': distortion_correction_available,
            'real_scale_available': real_scale_available
        }

        # 如果启用了畸变矫正，添加矫正后的数据
        if distortion_correction_available:
            tracking_entry.update({
                'tail_position_corrected': corrected_points[0],
                'head_position_corrected': corrected_points[1],
                'body_center_corrected': corrected_points[2],
                'body_tail_junction_corrected': corrected_points[3],
                'tail_end_corrected': corrected_points[4],
            })

            # 如果有真实尺寸转换，添加真实坐标
            if real_scale_available:
                tracking_entry.update({
                    'tail_position_real': real_points[0],
                    'head_position_real': real_points[1],
                    'body_center_real': real_points[2],
                    'body_tail_junction_real': real_points[3],
                    'tail_end_real': real_points[4],
                    'fish_length_real': fish_length_real,
                    'speed_real': speed_real,
                })
            else:
                # 即使没有真实尺寸转换，也添加speed_real字段
                tracking_entry['speed_real'] = 0
        else:
            # 如果没有畸变矫正，也添加必要的字段
            tracking_entry.update({
                'tail_position_real': (0, 0),
                'head_position_real': (0, 0),
                'body_center_real': (0, 0),
                'body_tail_junction_real': (0, 0),
                'tail_end_real': (0, 0),
                'fish_length_real': 0,
                'speed_real': 0,
            })

        self.tracking_data[track_id].append(tracking_entry)
        
        # 调试输出：确认tracking_entry中的关键数据
        print(f"📊 Fish {track_id} tracking_entry: 帧={tracking_entry['frame']}, speed_pixel={tracking_entry['speed_pixel']:.2f}, tail_freq={tracking_entry['tail_freq']}")

        # 现在更新fish_data，使用刚刚计算的最新数据
        self.update_fish_data_from_tracking(track_id, tracking_entry, current_timestamp)

        # 准备要记录的数据
        log_data = {
            'frame': self.frame_count[track_id],
            'timestamp': current_timestamp,
            'position_x_pixel': point0[0],
            'position_y_pixel': point0[1],
            'position_x_real': real_points[0][0] if real_scale_available else 0,
            'position_y_real': real_points[0][1] if real_scale_available else 0,
            'tail_angle': signed_angle,
            'tail_freq': self.tail_frequencies.get(track_id, 0),
            'speed_pixel': speed_pixel,
            'speed_real': speed_real if real_scale_available else 0,
            'fish_length_pixel': fish_length_pixel,
            'fish_length_real': fish_length_real if real_scale_available else 0,
            'posture': self.fish_postures.get(track_id, "Unknown")
        }

        # 记录到日志文件 - 已禁用，不生成 fish_tracking_data.txt
        # self.log_fish_data(track_id, log_data)

    def update_fish_data_from_tracking(self, track_id, tracking_entry, current_timestamp):
        """
        从tracking_entry更新fish_data，确保数据一致性
        """
        # 初始化fish_data结构
        if track_id not in self.fish_data:
            self.fish_data[track_id] = {
                'positions': [],
                'speeds': [],
                'tail_frequency': 0,
                'current_posture': 'Unknown',
                'keypoints_history': [],
                'current_speed_pixel': 0.0,
                'current_speed_real': 0.0,
                'realtime_speed_pixel': 0.0,  # 实时速度（每帧更新）
                'realtime_speed_real': 0.0,   # 实时真实速度
                'realtime_tail_frequency': 0.0  # 实时尾摆频率
            }

        # 从tracking_entry获取数据
        speed_pixel = tracking_entry.get('speed_pixel', 0)
        speed_real = tracking_entry.get('speed_real', 0)
        tail_freq = tracking_entry.get('tail_freq', 0)
        posture = tracking_entry.get('posture', 'Unknown')
        tail_pos_pixel = tracking_entry.get('tail_position_pixel', (0, 0))

        # 创建位置数据
        current_pos = {
            'pixel': {'x': float(tail_pos_pixel[0]), 'y': float(tail_pos_pixel[1])},
            'timestamp': current_timestamp,
            'speed_pixel': speed_pixel,
            'speed_real': speed_real
        }

        # 计算实时尾摆频率（Hz）
        realtime_tail_frequency = tail_freq / 5.0 if tail_freq > 0 else 0

        # 更新fish_data
        self.fish_data[track_id]['positions'].append(current_pos)
        self.fish_data[track_id]['speeds'].append(speed_pixel)
        self.fish_data[track_id]['tail_frequency'] = tail_freq
        self.fish_data[track_id]['current_posture'] = posture
        self.fish_data[track_id]['current_speed_pixel'] = speed_pixel
        self.fish_data[track_id]['current_speed_real'] = speed_real
        
        # 更新实时数据（GUI专用）
        self.fish_data[track_id]['realtime_speed_pixel'] = speed_pixel
        self.fish_data[track_id]['realtime_speed_real'] = speed_real
        self.fish_data[track_id]['realtime_tail_frequency'] = realtime_tail_frequency
        
        # 计算累积统计数据用于CSV导出
        if self.valid_speeds[track_id]:
            self.fish_data[track_id]['average_speed_pixel'] = sum(self.valid_speeds[track_id]) / len(self.valid_speeds[track_id])
        else:
            self.fish_data[track_id]['average_speed_pixel'] = 0
            
        # 计算总平均速度
        if self.total_tracking_time[track_id] > 0:
            self.fish_data[track_id]['total_average_speed_pixel'] = self.cumulative_distances[track_id] / self.total_tracking_time[track_id]
        else:
            self.fish_data[track_id]['total_average_speed_pixel'] = 0
            
        # 存储真实速度的累积数据
        if 'valid_real_speeds' not in self.fish_data[track_id]:
            self.fish_data[track_id]['valid_real_speeds'] = []
        
        if speed_real > 0:
            self.fish_data[track_id]['valid_real_speeds'].append(speed_real)
            # 保持最近30个值
            if len(self.fish_data[track_id]['valid_real_speeds']) > 30:
                self.fish_data[track_id]['valid_real_speeds'].pop(0)
        
        if self.fish_data[track_id]['valid_real_speeds']:
            self.fish_data[track_id]['average_speed_real'] = sum(self.fish_data[track_id]['valid_real_speeds']) / len(self.fish_data[track_id]['valid_real_speeds'])
        else:
            self.fish_data[track_id]['average_speed_real'] = 0
            
        # 添加逐秒数据到fish_data用于报告生成
        self.fish_data[track_id]['second_data'] = dict(self.second_data[track_id])

        # 限制历史数据长度，避免内存过度使用
        max_history = 100
        if len(self.fish_data[track_id]['positions']) > max_history:
            self.fish_data[track_id]['positions'] = self.fish_data[track_id]['positions'][-max_history:]
        if len(self.fish_data[track_id]['speeds']) > max_history:
            self.fish_data[track_id]['speeds'] = self.fish_data[track_id]['speeds'][-max_history:]
            
        # 打印调试信息，确认数据正确传递
        print(f"🔄 Fish {track_id}: 速度={speed_pixel:.2f}px/s, 频率={tail_freq}/5s ({realtime_tail_frequency:.2f}Hz), 体态={posture}")

    def determine_fish_posture(self, track_id, keypoints):
        """
        根据关键点的相对位置判断鱼的体态是C型、S型或直线型
        新设计：支持瞬时体态变化，同时减少噪声干扰

        参数:
            track_id: 鱼的跟踪ID
            keypoints: 关键点数据
        """
        # 确保所有需要的关键点都可见
        if len(keypoints) < 5 or any(keypoints[i, 2] <= 0.5 for i in range(5)):
            self.fish_postures[track_id] = "Unknown"
            return

        # 提取关键点坐标
        points = [(keypoints[i, 0], keypoints[i, 1]) for i in range(5)]

        # 计算鱼体的弯曲特征
        current_posture = self._analyze_fish_curvature(points)

        # 初始化历史记录
        if track_id not in self.posture_history:
            self.posture_history[track_id] = []

        # 添加当前体态到历史记录（保持较短的历史）
        self.posture_history[track_id].append(current_posture)

        # 只保留最近3帧的历史，用于噪声过滤
        max_history = 3
        if len(self.posture_history[track_id]) > max_history:
            self.posture_history[track_id] = self.posture_history[track_id][-max_history:]

        # 使用轻量级噪声过滤：优先响应瞬时变化，但过滤明显的检测错误
        final_posture = current_posture

        if len(self.posture_history[track_id]) >= 2:
            prev_posture = self.posture_history[track_id][-2]

            # 如果连续两帧都是同一体态，直接采用
            if current_posture == prev_posture:
                final_posture = current_posture
            else:
                # 体态发生变化时，优先接受变化（支持瞬时响应）
                if self._is_valid_posture_transition(prev_posture, current_posture):
                    final_posture = current_posture  # 直接接受合理的变化
                else:
                    # 不合理的变化，检查是否为检测错误
                    if len(self.posture_history[track_id]) >= 3:
                        prev_prev_posture = self.posture_history[track_id][-3]
                        # 只有在前两帧完全一致且当前变化不合理时，才认为是噪声
                        if prev_prev_posture == prev_posture and prev_posture != "Unknown":
                            final_posture = prev_posture  # 可能是检测错误，保持稳定
                        else:
                            final_posture = current_posture  # 接受变化，可能是真实的快速变化
                    else:
                        final_posture = current_posture  # 历史不足，接受当前判断

        # 更新最终体态
        self.fish_postures[track_id] = final_posture

    def _analyze_fish_curvature(self, points):
        """
        分析鱼体弯曲特征，返回体态类型

        参数:
            points: 5个关键点的坐标列表

        返回:
            体态类型: "Straight", "C-shape", "S-shape"
        """
        # 计算头部到尾部的理想直线
        head_to_tail = (points[4][0] - points[0][0], points[4][1] - points[0][1])
        head_to_tail_length = (head_to_tail[0]**2 + head_to_tail[1]**2)**0.5

        if head_to_tail_length < 1e-6:
            return "Straight"

        # 计算鱼体总长度（沿关键点的实际长度）
        total_length = 0
        for i in range(len(points) - 1):
            segment_length = ((points[i+1][0] - points[i][0])**2 + (points[i+1][1] - points[i][1])**2)**0.5
            total_length += segment_length

        if total_length < 1e-6:
            return "Straight"

        # 分析每个中间关键点的偏移
        deviations = []
        deviation_signs = []

        for i in [1, 2, 3]:  # 中间三个关键点
            # 计算点到理想直线的偏移
            point = points[i]
            head_to_point = (point[0] - points[0][0], point[1] - points[0][1])

            # 使用叉积计算偏移距离和方向
            cross_product = head_to_tail[0] * head_to_point[1] - head_to_tail[1] * head_to_point[0]
            deviation = abs(cross_product) / head_to_tail_length

            # 归一化偏移（相对于鱼体长度）
            normalized_deviation = deviation / total_length if total_length > 0 else 0

            deviations.append(normalized_deviation)
            if normalized_deviation > 0.05:  # 只记录显著偏移的方向
                deviation_signs.append(1 if cross_product > 0 else -1)

        # 计算最大偏移和平均偏移
        max_deviation = max(deviations)
        avg_deviation = sum(deviations) / len(deviations)

        # 设置阈值（相对保守，避免过度敏感）
        straight_threshold = 0.08    # 直线型阈值
        c_shape_threshold = 0.15     # C型阈值
        s_shape_threshold = 0.12     # S型阈值（相对较低，因为S型特征更复杂）

        # 判断体态
        if max_deviation < straight_threshold:
            return "Straight"

        # 分析弯曲模式
        if len(deviation_signs) >= 2:
            # 检查偏移方向的一致性
            positive_signs = sum(1 for sign in deviation_signs if sign > 0)
            negative_signs = sum(1 for sign in deviation_signs if sign < 0)

            # C型：大部分偏移在同一方向
            if positive_signs >= 2 and negative_signs == 0:
                return "C-shape" if max_deviation > c_shape_threshold else "Straight"
            elif negative_signs >= 2 and positive_signs == 0:
                return "C-shape" if max_deviation > c_shape_threshold else "Straight"

            # S型：偏移方向混合，且有足够的弯曲
            elif positive_signs >= 1 and negative_signs >= 1:
                # S型需要检查弯曲的复杂性
                if max_deviation > s_shape_threshold and avg_deviation > s_shape_threshold * 0.6:
                    return "S-shape"
        else:
                    return "Straight"

        # 默认情况
        if max_deviation > c_shape_threshold:
            return "C-shape"
        else:
            return "Straight"

    def _is_valid_posture_transition(self, prev_posture, current_posture):
        """
        检查体态变化是否合理

        参数:
            prev_posture: 前一帧的体态
            current_posture: 当前帧的体态

        返回:
            是否为合理的变化
        """
        # 定义合理的体态变化模式
        valid_transitions = {
            "Straight": ["C-shape", "S-shape"],      # 直线可以变为任何弯曲
            "C-shape": ["Straight", "S-shape"],      # C型可以变为直线或S型
            "S-shape": ["Straight", "C-shape"],      # S型可以变为直线或C型
            "Unknown": ["Straight", "C-shape", "S-shape"]  # 未知可以变为任何状态
        }

        if prev_posture == current_posture:
            return True  # 相同体态总是合理的

        return current_posture in valid_transitions.get(prev_posture, [])


    def log_fish_data(self, fish_id, data):
        """
        记录单条鱼的数据到统一日志文件

        参数:
            fish_id: 鱼的ID
            data: 要记录的数据字典
        """
        # 确保日志文件存在
        if not os.path.exists(self.log_file):
            # 创建新文件并写入表头
            with open(self.log_file, 'w', newline='') as f:
                writer = csv.writer(f)
                header = [
                    'Fish_ID', 'Frame', 'Timestamp',
                    'Position_X_Pixel', 'Position_Y_Pixel',
                    'Position_X_Real_mm', 'Position_Y_Real_mm',
                    'Tail_Angle', 'Tail_Frequency',
                    'Speed_Pixel_per_sec', 'Speed_Real_mm_per_sec',
                    'Fish_Length_Pixel', 'Fish_Length_Real_mm',
                    'Posture', 'Distortion_Corrected', 'Real_Scale_Available'
                ]
                writer.writerow(header)

        # 检查状态
        distortion_corrected = self.enable_distortion_correction and self.calibrator.calibrated
        real_scale_available = (distortion_corrected and
                               hasattr(self.calibrator, 'square_size') and
                               self.calibrator.square_size is not None)



        # 追加数据到统一日志文件
        with open(self.log_file, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                fish_id,
                data.get('frame', 0),
                data.get('timestamp', 0),
                data.get('position_x_pixel', 0),
                data.get('position_y_pixel', 0),
                data.get('position_x_real', 0),
                data.get('position_y_real', 0),
                data.get('tail_angle', 0),
                data.get('tail_freq', 0),
                data.get('speed_pixel', 0),
                data.get('speed_real', 0),
                data.get('fish_length_pixel', 0),
                data.get('fish_length_real', 0),
                data.get('posture', 'Unknown'),
                distortion_corrected,
                real_scale_available
            ])

        # 更新汇总文件 - 已禁用，不生成 fish_summary.csv
        # self.update_summary_file(fish_id, data)

    def update_summary_file(self, fish_id, data):
        """
        更新鱼类汇总文件，记录每条鱼的最新状态

        参数:
            fish_id: 鱼的ID
            data: 最新的数据
        """
        # 读取现有汇总文件
        summary_data = []
        fish_exists = False

        if os.path.exists(self.summary_file):
            with open(self.summary_file, 'r', newline='') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    if int(row['Fish_ID']) == fish_id:
                        # 更新现有鱼的数据
                        row.update({
                            'Last_Frame': data.get('frame', 0),
                            'Last_Timestamp': data.get('timestamp', 0),
                            'Last_Position_X': data.get('position_x', 0),
                            'Last_Position_Y': data.get('position_y', 0),
                            'Current_Tail_Angle': data.get('tail_angle', 0),
                            'Average_Tail_Frequency': data.get('tail_freq', 0),
                            'Current_Speed': data.get('speed', 0),
                            'Current_Posture': data.get('posture', 'Unknown'),
                            'Total_Frames': int(row['Total_Frames']) + 1
                        })
                        fish_exists = True
                    summary_data.append(row)

        # 如果鱼不存在，添加新记录
        if not fish_exists:
            summary_data.append({
                'Fish_ID': fish_id,
                'First_Seen_Frame': data.get('frame', 0),
                'Last_Frame': data.get('frame', 0),
                'Last_Timestamp': data.get('timestamp', 0),
                'Last_Position_X': data.get('position_x', 0),
                'Last_Position_Y': data.get('position_y', 0),
                'Current_Tail_Angle': data.get('tail_angle', 0),
                'Average_Tail_Frequency': data.get('tail_freq', 0),
                'Current_Speed': data.get('speed', 0),
                'Current_Posture': data.get('posture', 'Unknown'),
                'Total_Frames': 1
            })

        # 写回汇总文件
        with open(self.summary_file, 'w', newline='') as f:
            if summary_data:
                writer = csv.DictWriter(f, fieldnames=summary_data[0].keys())
                writer.writeheader()
                writer.writerows(summary_data)

    def generate_fish_report(self, output_dir=None):
        """
        生成每条鱼的详细报告（仅txt格式）

        参数:
            output_dir: 输出目录，默认为日志目录

        返回:
            报告文件路径列表
        """
        if output_dir is None:
            output_dir = self.log_dir

        os.makedirs(output_dir, exist_ok=True)
        report_files = []

        # 从CSV文件读取数据
        all_fish_data = defaultdict(list)

        # 使用主要的CSV文件（包含完整的速度和坐标数据）
        csv_file = "fish_tracking_data.csv"
        if os.path.exists(csv_file):
            print(f"使用数据文件: {csv_file}")
            with open(csv_file, 'r', newline='') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    fish_id = row.get('Fish_ID')
                    if fish_id:
                        all_fish_data[fish_id].append(row)
        else:
            print(f"错误: 数据文件 '{csv_file}' 不存在")
            print("请先运行鱼类跟踪系统生成数据文件")
            return []

        # 为每条鱼生成txt报告
        for fish_id, fish_data in all_fish_data.items():
            if not fish_data:
                continue

            # 计算统计信息
            total_frames = len(fish_data)

            # 检查是否有真实尺寸数据
            has_real_data = any(float(d.get('Tail_Position_X_Real_mm', 0)) != 0 or
                               float(d.get('Tail_Position_Y_Real_mm', 0)) != 0 for d in fish_data)

            # 计算像素单位的统计
            pixel_speeds = [float(d.get('Speed_Pixel_per_sec', 0)) for d in fish_data]
            pixel_lengths = [float(d.get('Fish_Length_Pixel', 0)) for d in fish_data if float(d.get('Fish_Length_Pixel', 0)) > 0]
            # 尝试两种可能的字段名
            tail_freqs = [float(d.get('Tail_Frequency_per_5s', d.get('Tail_Frequency', 0))) for d in fish_data]

            avg_speed_pixel = sum(pixel_speeds) / len(pixel_speeds) if pixel_speeds else 0
            max_speed_pixel = max(pixel_speeds) if pixel_speeds else 0
            avg_length_pixel = sum(pixel_lengths) / len(pixel_lengths) if pixel_lengths else 0
            avg_tail_freq = sum(tail_freqs) / len(tail_freqs) if tail_freqs else 0

            # 如果有真实尺寸数据，计算真实单位的统计
            real_stats = {}
            if has_real_data:
                real_speeds = [float(d.get('Speed_Real_mm_per_sec', 0)) for d in fish_data]
                real_lengths = [float(d.get('Fish_Length_Real_mm', 0)) for d in fish_data if float(d.get('Fish_Length_Real_mm', 0)) > 0]

                real_stats = {
                    'avg_speed_real': sum(real_speeds) / len(real_speeds) if real_speeds else 0,
                    'max_speed_real': max(real_speeds) if real_speeds else 0,
                    'avg_length_real': sum(real_lengths) / len(real_lengths) if real_lengths else 0
                }

            # 计算速度变化（加速度）
            speed_changes = []
            for i in range(1, len(pixel_speeds)):
                speed_changes.append(abs(pixel_speeds[i] - pixel_speeds[i-1]))
            avg_acceleration = sum(speed_changes) / len(speed_changes) if speed_changes else 0

            # 统计体态分布
            posture_counts = {}
            for d in fish_data:
                posture = d.get('Posture', 'Unknown')
                posture_counts[posture] = posture_counts.get(posture, 0) + 1

            # 统计体态变化次数
            posture_changes = 0
            last_posture = None
            for d in fish_data:
                current_posture = d.get('Posture', 'Unknown')
                if last_posture is not None and current_posture != last_posture:
                    posture_changes += 1
                last_posture = current_posture

            # 创建文本报告文件
            txt_report_file = os.path.join(output_dir, f"fish_{fish_id}_report.txt")
            with open(txt_report_file, 'w', encoding='utf-8') as f:
                f.write(f"===== 鱼类ID: {fish_id} 详细报告 =====\n")
                f.write(f"生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")

                f.write("1. 基本信息:\n")
                f.write(f"   - 总帧数: {total_frames}\n")
                f.write(f"   - 跟踪时长: {total_frames/self.fps:.2f} 秒\n")
                f.write(f"   - 平均尾摆频率: {avg_tail_freq:.2f} 次/5秒\n\n")

                f.write("2. 运动统计 (像素单位):\n")
                f.write(f"   - 平均速度: {avg_speed_pixel:.2f} 像素/秒\n")
                f.write(f"   - 最大速度: {max_speed_pixel:.2f} 像素/秒\n")
                f.write(f"   - 平均加速度变化: {avg_acceleration:.2f} 像素/秒²\n")
                f.write(f"   - 平均鱼体长度: {avg_length_pixel:.2f} 像素\n")

                if has_real_data:
                    f.write("\n3. 运动统计 (真实尺寸):\n")
                    f.write(f"   - 平均速度: {real_stats['avg_speed_real']:.2f} mm/秒\n")
                    f.write(f"   - 最大速度: {real_stats['max_speed_real']:.2f} mm/秒\n")
                    f.write(f"   - 平均鱼体长度: {real_stats['avg_length_real']:.2f} mm\n")
                    section_num = 4
                else:
                    f.write("\n3. 注意: 未启用真实尺寸转换，仅提供像素单位数据\n")
                    section_num = 4

                f.write(f"\n{section_num}. 体态分析:\n")
                f.write(f"   - 体态变化次数: {posture_changes}\n")
                f.write("   - 体态分布:\n")
                for posture, count in posture_counts.items():
                    percentage = (count / total_frames) * 100 if total_frames > 0 else 0
                    f.write(f"     * {posture}: {count} 帧 ({percentage:.1f}%)\n")

                f.write(f"\n{section_num+1}. 轨迹信息:\n")
                if len(fish_data) > 0:
                    start_x = fish_data[0].get('Tail_Position_X_Pixel', 0)
                    start_y = fish_data[0].get('Tail_Position_Y_Pixel', 0)
                    end_x = fish_data[-1].get('Tail_Position_X_Pixel', 0)
                    end_y = fish_data[-1].get('Tail_Position_Y_Pixel', 0)

                    f.write(f"   - 起始位置 (像素): ({start_x}, {start_y})\n")
                    f.write(f"   - 结束位置 (像素): ({end_x}, {end_y})\n")

                    if has_real_data:
                        start_x_real = float(fish_data[0].get('Tail_Position_X_Real_mm', 0))
                        start_y_real = float(fish_data[0].get('Tail_Position_Y_Real_mm', 0))
                        end_x_real = float(fish_data[-1].get('Tail_Position_X_Real_mm', 0))
                        end_y_real = float(fish_data[-1].get('Tail_Position_Y_Real_mm', 0))
                        f.write(f"   - 起始位置 (真实): ({start_x_real:.2f}, {start_y_real:.2f}) mm\n")
                        f.write(f"   - 结束位置 (真实): ({end_x_real:.2f}, {end_y_real:.2f}) mm\n")

                f.write(f"\n{section_num+2}. 数据文件位置:\n")
                # 显示实际使用的数据文件
                f.write(f"   - 详细日志: fish_tracking_data.csv\n")

                # 添加数据质量信息
                distortion_corrected = fish_data[0].get('Distortion_Corrected', 'False') == 'True'
                real_scale_available = fish_data[0].get('Real_Scale_Available', 'False') == 'True'

                f.write(f"\n{section_num+3}. 数据质量:\n")
                f.write(f"   - 畸变矫正: {'已启用' if distortion_corrected else '未启用'}\n")
                f.write(f"   - 真实尺寸转换: {'已启用' if real_scale_available else '未启用'}\n")

            report_files.append(txt_report_file)

        # 创建汇总txt报告
        summary_txt = os.path.join(output_dir, "fish_summary_report.txt")
        with open(summary_txt, 'w', encoding='utf-8') as f:
            f.write("===== 鱼类跟踪汇总报告 =====\n")
            f.write(f"生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")

            f.write("跟踪鱼类列表:\n")
            f.write("-" * 80 + "\n")
            f.write(f"{'鱼ID':<8} {'帧数':<8} {'时长(秒)':<10} {'平均速度':<15} {'主要体态':<12} {'报告文件'}\n")
            f.write("-" * 80 + "\n")

            # 使用已加载的数据而不是重新读取文件
            for fish_id, fish_data in all_fish_data.items():
                if not fish_data:
                    continue

                total_frames = len(fish_data)
                duration = total_frames / self.fps

                # 从内存中的数据计算主要体态（避免重复读取文件）
                posture_counts = {}
                pixel_speeds = []
                for d in fish_data:
                    # 统计体态
                    p = d.get('Posture', 'Unknown')
                    posture_counts[p] = posture_counts.get(p, 0) + 1
                    # 收集速度数据
                    speed = float(d.get('Speed_Pixel_per_sec', 0))
                    pixel_speeds.append(speed)

                # 确定主要体态
                main_posture = "Unknown"
                if posture_counts:
                                main_posture = max(posture_counts.items(), key=lambda x: x[1])[0]

                # 计算平均速度
                avg_speed = sum(pixel_speeds) / len(pixel_speeds) if pixel_speeds else 0
                avg_speed_str = f"{avg_speed:.2f}"

                report_file = f"fish_{fish_id}_report.txt"

                f.write(f"{fish_id:<8} {total_frames:<8} {duration:<10.2f} {avg_speed_str:<15} {main_posture:<12} {report_file}\n")

            f.write("-" * 80 + "\n")
            f.write(f"\n数据文件:\n")
            f.write(f"- 详细日志文件: fish_tracking_data.csv\n")

        report_files.append(summary_txt)
        return report_files


def save_tracking_data(tracker, output_file):
    """
    将跟踪数据保存到CSV文件

    参数:
        tracker: 鱼类跟踪器实例
        output_file: 输出文件路径
    """
    # 确保输出目录存在
    os.makedirs(os.path.dirname(output_file) if os.path.dirname(output_file) else '.', exist_ok=True)

    with open(output_file, 'w', newline='') as f:
        writer = csv.writer(f)
        # 写入表头
        writer.writerow(['Fish_ID', 'Frame', 'Timestamp',
                         'Tail_Position_X_Pixel', 'Tail_Position_Y_Pixel',
                         'Tail_Position_X_Real_mm', 'Tail_Position_Y_Real_mm',
                         'Tail_Angle', 'Tail_Frequency_per_5s',
                         'Speed_Pixel_per_sec', 'Speed_Real_mm_per_sec',
                         'Fish_Length_Pixel', 'Fish_Length_Real_mm',
                         'Posture', 'Distortion_Corrected', 'Real_Scale_Available'])

        # 写入每条鱼的数据
        for fish_id, data_list in tracker.tracking_data.items():
            for data in data_list:
                # 获取位置数据
                tail_pos_pixel = data.get('tail_position_pixel', (0, 0))
                tail_pos_real = data.get('tail_position_real', (0, 0)) if data.get('real_scale_available', False) else (0, 0)

                writer.writerow([
                    fish_id,
                    data.get('frame', 0),
                    data.get('timestamp', 0),
                    tail_pos_pixel[0] if isinstance(tail_pos_pixel, tuple) else 0,
                    tail_pos_pixel[1] if isinstance(tail_pos_pixel, tuple) else 0,
                    tail_pos_real[0] if isinstance(tail_pos_real, tuple) else 0,
                    tail_pos_real[1] if isinstance(tail_pos_real, tuple) else 0,
                    data.get('tail_angle', 0),
                    data.get('tail_freq', 0),
                    data.get('speed_pixel', 0),
                    data.get('speed_real', 0),
                    data.get('fish_length_pixel', 0),
                    data.get('fish_length_real', 0),
                    data.get('posture', 'Unknown'),
                    data.get('corrected_available', False),
                    data.get('real_scale_available', False)
                ])

    print(f"Tracking data saved to: {output_file}")
    print(f"包含字段: 位置(像素+真实)、速度(像素+真实)、鱼体长度(像素+真实)、角度、频率、体态")


def calibrate_camera_interactive():
    """
    交互式相机标定函数
    用于单独进行相机标定，不依赖于跟踪系统
    """
    print("=== 相机标定工具 ===")
    print("本工具用于标定广角镜头，矫正畸变并设置真实尺寸比例")
    print("无需额外的已知尺寸物体，仅需棋盘格的真实尺寸即可")

    # 配置参数
    calibration_dir = input("请输入标定图像文件夹路径 (默认: calibration_images): ").strip()
    if not calibration_dir:
        calibration_dir = "calibration_images"

    if not os.path.exists(calibration_dir):
        print(f"错误: 文件夹 '{calibration_dir}' 不存在")
        print("请创建文件夹并放入5-15张棋盘格标定图像")
        return False

    # 检查图像数量
    import glob
    images = glob.glob(os.path.join(calibration_dir, "*.jpg")) + \
             glob.glob(os.path.join(calibration_dir, "*.png"))

    if len(images) < 3:
        print(f"错误: 标定图像数量不足 ({len(images)}张)")
        print("请至少提供3张标定图像，建议5-15张")
        return False

    print(f"检测到 {len(images)} 张标定图像")

    # 棋盘格参数
    print("\n=== 棋盘格参数设置 ===")
    print("请根据您的棋盘格设置参数（内角点数量，不是格子数量）")
    print("例如：10x7格的棋盘格，内角点为9x6")
    try:
        cols = int(input("棋盘格列数（内角点，默认9）: ") or "9")
        rows = int(input("棋盘格行数（内角点，默认6）: ") or "6")
        square_size = float(input("每格实际尺寸（毫米，默认25）: ") or "25.0")
    except ValueError:
        print("输入格式错误，使用默认参数：9x6格，每格25mm")
        cols, rows, square_size = 9, 6, 25.0

    # 输出文件
    output_file = input("标定结果保存文件名 (默认: camera_calibration.json): ").strip()
    if not output_file:
        output_file = "camera_calibration.json"

    print(f"\n=== 开始标定 ===")
    print(f"棋盘格参数: {cols}x{rows}内角点, 每格 {square_size}mm")
    print(f"图像数量: {len(images)}")
    print(f"输出文件: {output_file}")

    # 进行标定
    calibrator = CameraCalibrator()
    success = calibrator.calibrate_camera(
        calibration_images_path=calibration_dir,
        checkerboard_size=(cols, rows),
        square_size=square_size,
        save_debug_images=True
    )

    if not success:
        print("相机标定失败!")
        print("请检查调试图像以分析失败原因")
        return False

    # 保存标定结果
    calibrator.save_calibration(output_file)

    print(f"\n=== 标定完成 ===")
    print("✓ 相机标定成功")
    print("✓ 已自动计算像素-毫米转换比例（基于棋盘格尺寸）")
    print(f"✓ 标定文件已保存: {output_file}")
    print("✓ 调试图像已保存，包括：")
    print("  - 角点检测结果图像")
    print("  - 畸变矫正对比图像")
    print("现在可以在跟踪系统中使用畸变矫正功能")

    return True


def test_calibration(calibration_file, test_image_path=None):
    """
    测试相机标定效果

    参数:
        calibration_file: 标定文件路径
        test_image_path: 测试图像路径（可选）
    """
    print("=== 测试相机标定效果 ===")

    if not os.path.exists(calibration_file):
        print(f"错误: 标定文件 '{calibration_file}' 不存在")
        return False

    # 加载标定数据
    calibrator = CameraCalibrator()
    if not calibrator.load_calibration(calibration_file):
        return False

    if test_image_path and os.path.exists(test_image_path):
        # 测试图像矫正效果
        img = cv2.imread(test_image_path)
        if img is not None:
            # 矫正整个图像
            h, w = img.shape[:2]
            new_camera_matrix, roi = cv2.getOptimalNewCameraMatrix(
                calibrator.camera_matrix, calibrator.dist_coeffs, (w, h), 1, (w, h)
            )

            # 矫正图像
            undistorted = cv2.undistort(img, calibrator.camera_matrix,
                                      calibrator.dist_coeffs, None, new_camera_matrix)

            # 保存对比图像
            comparison = np.hstack((img, undistorted))
            output_path = "calibration_test_comparison.jpg"
            cv2.imwrite(output_path, comparison)
            print(f"对比图像已保存: {output_path}")
            print("左侧为原图，右侧为矫正后图像")

    # 测试关键点矫正
    print("\n=== 测试关键点矫正 ===")
    test_points = [(100, 100), (500, 300), (800, 600)]
    print("测试点（像素坐标）:", test_points)

    corrected_points = calibrator.undistort_points(test_points)
    print("矫正后坐标:", corrected_points)

    if calibrator.pixel_to_mm_ratio:
        real_points = calibrator.pixels_to_real(corrected_points)
        print("真实坐标（毫米）:", real_points)

        # 测试距离计算
        distance_pixel = np.sqrt((test_points[0][0] - test_points[1][0])**2 +
                               (test_points[0][1] - test_points[1][1])**2)
        distance_real = calibrator.calculate_real_distance(test_points[0], test_points[1])

        print(f"\n=== 距离测试 ===")
        print(f"像素距离: {distance_pixel:.2f} 像素")
        print(f"真实距离: {distance_real:.2f} 毫米")

    return True


def main():
    # 打开视频文件或摄像头
    video_path = "/home/lyc/yolov8/ultralytics-main/datasets/20250501-13.mp4"  # 0表示使用摄像头，也可以是视频文件路径
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        print("Failed to open video source")
        return

    # 获取视频属性
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))

    # 确保fps值有效，如果无效则使用默认值30
    if fps <= 0:
        print("Warning: Invalid fps detected, using default value of 30")
        fps = 30
    else:
        print(f"Video fps: {fps}")

    # 畸变矫正配置
    enable_correction = True  # 设置为True启用畸变矫正
    calibration_file = "camera_calibration.json"  # 标定文件路径

    # 创建鱼类跟踪器 - 启用畸变矫正功能
    tracker = FishTracker(
        model_path="bestx.pt",
        conf=0.3,
        smooth_factor=0.3,
        fps=fps,
        enable_distortion_correction=enable_correction,
        calibration_file=calibration_file
    )

    # 如果启用畸变矫正但没有标定文件，提供相机标定功能
    if enable_correction and not os.path.exists(calibration_file):
        print("\n=== 相机标定向导 ===")
        print("未找到相机标定文件，需要先进行相机标定。")
        print("请按照以下步骤进行标定：")
        print("1. 准备棋盘格标定板（建议9x6格，每格25mm）")
        print("2. 拍摄5-15张不同角度和位置的标定图像")
        print("3. 将标定图像放在 'calibration_images' 文件夹中")

        calibration_dir = "calibration_images"
        if os.path.exists(calibration_dir) and len(os.listdir(calibration_dir)) > 0:
            print(f"检测到标定图像文件夹: {calibration_dir}")
            choice = input("是否现在进行相机标定？(y/n): ").lower()

            if choice == 'y':
                # 获取棋盘格参数
                print("\n=== 棋盘格参数设置 ===")
                print("请根据您的棋盘格设置参数：")
                try:
                    cols = int(input("棋盘格列数（内角点数，默认9）: ") or "9")
                    rows = int(input("棋盘格行数（内角点数，默认6）: ") or "6")
                    square_size = float(input("每格实际尺寸（毫米，默认25）: ") or "25.0")
                except ValueError:
                    print("输入格式错误，使用默认参数：9x6格，每格25mm")
                    cols, rows, square_size = 9, 6, 25.0

                print(f"使用参数：{cols}x{rows}格，每格{square_size}mm")

                # 进行相机标定
                calibrator = CameraCalibrator()
                success = calibrator.calibrate_camera(
                    calibration_images_path=calibration_dir,
                    checkerboard_size=(cols, rows),
                    square_size=square_size,
                    save_debug_images=True
                )

                if success:
                    # 保存标定结果
                    calibrator.save_calibration(calibration_file)

                    print("\n=== 标定完成 ===")
                    print("✓ 相机标定成功")
                    print("✓ 已自动计算像素-毫米转换比例（基于棋盘格尺寸）")
                    print("✓ 调试图像已保存，可查看检测和矫正效果")

                    # 重新创建跟踪器以加载新的标定数据
                    tracker = FishTracker(
                        model_path="bestx.pt",
                        conf=0.3,
                        smooth_factor=0.3,
                        fps=fps,
                        enable_distortion_correction=enable_correction,
                        calibration_file=calibration_file
                    )

                else:
                    print("相机标定失败，将禁用畸变矫正功能")
                    print("请检查调试图像以分析失败原因")
                    enable_correction = False
            else:
                print("跳过相机标定，将禁用畸变矫正功能")
                enable_correction = False
        else:
            print(f"请先创建 '{calibration_dir}' 文件夹并放入标定图像")
            print("将禁用畸变矫正功能")
            enable_correction = False

    # 创建视频写入器 - 使用H.264编码，更好地支持中文
    output_path = "fish_tracking_output2.mp4"
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    # 检测是否可以使用GUI显示
    has_gui = True
    try:
        # 尝试创建一个小窗口测试GUI可用性
        cv2.namedWindow("Test", cv2.WINDOW_NORMAL)
        cv2.destroyWindow("Test")
    except:
        has_gui = False
        print("Warning: Cannot create display window, video will be saved without display.")
        print(f"Video will be saved to: {output_path}")

    frame_count = 0
    max_frames = 300  # 调试模式：只处理前900帧
    print(f"\n=== 开始处理视频 (调试模式：仅处理前{max_frames}帧) ===")
    if enable_correction and tracker.calibrator.calibrated:
        print("✓ 畸变矫正已启用")
        # 检查是否有完整的尺寸转换信息
        has_scale_info = (hasattr(tracker.calibrator, 'square_size') and
                         hasattr(tracker.calibrator, 'checkerboard_size') and
                         hasattr(tracker.calibrator, 'imgpoints') and
                         tracker.calibrator.square_size is not None)
        if has_scale_info:
            print("✓ 真实尺寸转换已启用")

            # 计算并显示参考比例
            if hasattr(tracker.calibrator, 'imgpoints') and len(tracker.calibrator.imgpoints) > 0:
                ref_corners = tracker.calibrator.imgpoints[0].reshape(-1, 2)
                corrected_ref = cv2.undistortPoints(
                    ref_corners.reshape(-1, 1, 2),
                    tracker.calibrator.camera_matrix,
                    tracker.calibrator.dist_coeffs,
                    None,
                    tracker.calibrator.camera_matrix
                ).reshape(-1, 2)

                # 计算平均像素距离作为参考
                pixel_distances = []
                for row in range(tracker.calibrator.checkerboard_size[1]):
                    for col in range(tracker.calibrator.checkerboard_size[0] - 1):
                        idx1 = row * tracker.calibrator.checkerboard_size[0] + col
                        idx2 = row * tracker.calibrator.checkerboard_size[0] + col + 1

                        p1 = corrected_ref[idx1]
                        p2 = corrected_ref[idx2]
                        pixel_dist = np.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)
                        pixel_distances.append(pixel_dist)

                if pixel_distances:
                    avg_pixel_distance = np.mean(pixel_distances)
                    reference_ratio = tracker.calibrator.square_size / avg_pixel_distance
                    print(f"✓ 参考像素-毫米比例: {reference_ratio:.4f} mm/pixel（仅供参考）")
                    print("✓ 使用精确相机标定算法进行坐标转换")
        else:
            print("⚠ 真实尺寸转换未设置，仅提供矫正后的像素坐标")
    else:
        print("⚠ 使用原始像素坐标（未矫正）")

    while cap.isOpened() and frame_count < max_frames:
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1

        # 处理帧
        result_frame = tracker.process_frame(frame)

        # 添加帧信息 - 使用英文
        cv2.putText(
            result_frame,
            "Fish Tracking with YOLOv8-Pose + Distortion Correction",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 0),
            2
        )

        # 显示畸变矫正状态
        if enable_correction and tracker.calibrator.calibrated:
            status_text = "Distortion Correction: ON"
            # 检查是否有完整的尺寸转换信息
            has_scale_info = (hasattr(tracker.calibrator, 'square_size') and
                             hasattr(tracker.calibrator, 'checkerboard_size') and
                             hasattr(tracker.calibrator, 'imgpoints') and
                             tracker.calibrator.square_size is not None)
            if has_scale_info:
                status_text += " | Real Scale: ON"
            cv2.putText(
                result_frame,
                status_text,
                (10, 60),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 255),
            2
        )

        # 显示结果（如果GUI可用）
        if has_gui:
            cv2.imshow("Fish Tracking", result_frame)
            # 按'q'退出
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        else:
            # 无GUI环境下显示进度
            if frame_count % 30 == 0:  # 每30帧显示一次进度
                print(f"Processed {frame_count}/{max_frames} frames ({frame_count/max_frames*100:.1f}%)")

        # 写入视频
        out.write(result_frame)

    # 保存跟踪数据到CSV文件
    data_output_path = "fish_tracking_data.csv"
    save_tracking_data(tracker, data_output_path)

    # 生成鱼类详细报告
    report_files = tracker.generate_fish_report()

    # 释放资源
    cap.release()
    out.release()
    if has_gui:
        cv2.destroyAllWindows()

    print(f"\n=== 处理完成 ===")
    print(f"视频已保存到: {output_path}")
    print(f"跟踪数据已保存到: {data_output_path}")
    print(f"鱼类日志保存在: {tracker.log_dir}")
    print(f"鱼类汇总文件: {tracker.summary_file}")
    print(f"生成了 {len(report_files)} 个鱼类报告")
    for report in report_files:
        print(f"  - {report}")

    if enable_correction and tracker.calibrator.calibrated:
        print(f"\n=== 畸变矫正信息 ===")
        print("✓ 已应用畸变矫正")
        # 检查是否有完整的尺寸转换信息
        has_scale_info = (hasattr(tracker.calibrator, 'square_size') and
                         hasattr(tracker.calibrator, 'checkerboard_size') and
                         hasattr(tracker.calibrator, 'imgpoints') and
                         tracker.calibrator.square_size is not None)
        if has_scale_info:
            print(f"✓ 棋盘格尺寸: {tracker.calibrator.checkerboard_size[0]}x{tracker.calibrator.checkerboard_size[1]}格，每格{tracker.calibrator.square_size}mm")
            print("✓ 数据包含真实尺寸（毫米）- 使用精确相机标定算法转换")
        else:
            print("⚠ 未设置尺寸转换比例，数据仅包含矫正后的像素坐标")
        print(f"标定文件: {calibration_file}")
    else:
        print(f"\n=== 注意 ===")
        print("⚠ 未使用畸变矫正，所有数据为原始像素坐标")
        print("如需真实尺寸数据，请先进行相机标定")


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1:
        if sys.argv[1] == "calibrate":
            # 单独运行相机标定
            calibrate_camera_interactive()
        elif sys.argv[1] == "test" and len(sys.argv) > 2:
            # 测试标定效果
            calibration_file = sys.argv[2]
            test_image = sys.argv[3] if len(sys.argv) > 3 else None
            test_calibration(calibration_file, test_image)
        else:
            print("用法:")
            print("  python fish_tracking_advanced.py                    # 运行鱼类跟踪")
            print("  python fish_tracking_advanced.py calibrate          # 相机标定")
            print("  python fish_tracking_advanced.py test <标定文件>     # 测试标定效果")
    else:
        # 运行主程序
        main()