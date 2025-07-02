import sys
import os
import cv2
import numpy as np
import json
import threading
import time
from datetime import datetime
from collections import defaultdict, deque
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
import seaborn as sns

from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                            QHBoxLayout, QGridLayout, QPushButton, QLabel, 
                            QComboBox, QLineEdit, QTextEdit, QFileDialog, 
                            QProgressBar, QTabWidget, QTableWidget, QTableWidgetItem,
                            QSlider, QCheckBox, QSpinBox, QDoubleSpinBox,
                            QGroupBox, QSplitter, QScrollArea, QFrame,
                            QMessageBox, QDialog, QDialogButtonBox, QRadioButton, QButtonGroup,
                            QProgressDialog)
from PyQt5.QtCore import Qt, QTimer, QThread, pyqtSignal, QMutex, QMutexLocker
from PyQt5.QtGui import QPixmap, QImage, QFont, QPalette, QColor, QIcon

from fish_tracking_advanced import FishTracker, CameraCalibrator


class VideoThread(QThread):
    """视频处理线程"""
    frame_ready = pyqtSignal(np.ndarray)
    tracking_data_ready = pyqtSignal(dict)
    error_occurred = pyqtSignal(str)

    def __init__(self):
        super().__init__()
        self.tracker = None
        self.video_source = None
        self.is_running = False
        self.mutex = QMutex()
        self.processed_frames = []  # 存储处理后的帧
        self.original_fps = 30  # 默认30fps
        
    def setup_tracker(self, model_path, tracker_type, conf_threshold, 
                     enable_correction, calibration_file, batch_mode=False,
                     use_cuda=True, save_interval=1.0):
        """设置跟踪器"""
        try:
            self.tracker = FishTracker(
                model_path=model_path,
                tracker=tracker_type,
                conf=conf_threshold,
                enable_distortion_correction=enable_correction,
                calibration_file=calibration_file,
                batch_mode=batch_mode,
                use_cuda=use_cuda,
                save_interval=save_interval
            )
            return True
        except Exception as e:
            self.error_occurred.emit(f"跟踪器设置失败: {str(e)}")
            return False
    
    def set_video_source(self, source):
        """设置视频源"""
        self.video_source = source
        
    def run(self):
        """主处理循环"""
        if not self.tracker or not self.video_source:
            self.error_occurred.emit("跟踪器或视频源未设置")
            return
            
        try:
            if isinstance(self.video_source, str):
                cap = cv2.VideoCapture(self.video_source)
            else:
                cap = cv2.VideoCapture(self.video_source)  # 摄像头
                
            if not cap.isOpened():
                self.error_occurred.emit("无法打开视频源")
                return
                
            # 获取原视频帧率
            self.original_fps = cap.get(cv2.CAP_PROP_FPS)
            if self.original_fps <= 0:
                self.original_fps = 30  # 如果无法获取，使用默认值
            
            # 更新tracker的fps为实际视频fps
            if self.tracker and hasattr(self.tracker, 'set_fps'):
                self.tracker.set_fps(self.original_fps)
                
            self.is_running = True
            self.processed_frames.clear()  # 清空之前的帧
            
            # 计算帧间隔（毫秒）
            frame_interval = int(1000 / self.original_fps) if self.original_fps > 0 else 33
            
            while self.is_running:
                ret, frame = cap.read()
                if not ret:
                    break
                    
                with QMutexLocker(self.mutex):
                    if not self.is_running:
                        break
                        
                # 处理帧
                annotated_frame = self.tracker.process_frame(frame)
                
                # 存储处理后的帧
                self.processed_frames.append(annotated_frame.copy())
                
                # 发送帧和跟踪数据
                self.frame_ready.emit(annotated_frame)
                
                # 获取当前跟踪数据
                tracking_data = self._get_current_tracking_data()
                self.tracking_data_ready.emit(tracking_data)
                
                # 使用原视频帧率
                self.msleep(frame_interval)
                
        except Exception as e:
            self.error_occurred.emit(f"视频处理错误: {str(e)}")
        finally:
            if 'cap' in locals():
                cap.release()
                
    def stop(self):
        """停止处理"""
        with QMutexLocker(self.mutex):
            self.is_running = False
        self.wait()
        
    def _get_current_tracking_data(self):
        """获取当前跟踪数据"""
        if not self.tracker:
            return {}
        
        # 获取过滤后的鱼类数据
        if hasattr(self.tracker, 'get_filtered_fish_data'):
            filter_enabled = getattr(self, 'filter_enabled_check', None)
            if filter_enabled and filter_enabled.isChecked():
                # 获取过滤参数
                filter_params = {
                    'min_track_length': getattr(self, 'min_track_length_spin', None).value() if hasattr(self, 'min_track_length_spin') else 10,
                    'min_tail_frequency': getattr(self, 'min_freq_spin', None).value() if hasattr(self, 'min_freq_spin') else 0.1,
                    'min_speed': getattr(self, 'min_speed_spin', None).value() if hasattr(self, 'min_speed_spin') else 1.0
                }
                fish_data = self.tracker.get_filtered_fish_data(filter_invalid=True, **filter_params)
            else:
                fish_data = self.tracker.fish_data
        else:
            fish_data = self.tracker.fish_data
            
        data = {
            'fish_count': len(fish_data),
            'total_fish_count': len(self.tracker.fish_data),
            'fish_info': {},
            'global_stats': self.tracker.global_stats
        }
        
        for fish_id, fish_info in fish_data.items():
            if fish_info['positions']:
                latest_pos = fish_info['positions'][-1]
                
                # 优先使用实时数据，如果没有则使用原有数据
                realtime_speed_pixel = fish_info.get('realtime_speed_pixel', 0)
                realtime_speed_real = fish_info.get('realtime_speed_real', 0) 
                realtime_tail_frequency = fish_info.get('realtime_tail_frequency', 0)
                
                # 如果实时数据为0，尝试使用其他数据源
                if realtime_speed_pixel == 0:
                    realtime_speed_pixel = fish_info.get('current_speed_pixel', 0)
                if realtime_speed_real == 0:
                    realtime_speed_real = fish_info.get('current_speed_real', 0)
                if realtime_tail_frequency == 0:
                    # 尾摆频率从5秒计数转换为Hz
                    tail_count_5s = fish_info.get('tail_frequency', 0)
                    realtime_tail_frequency = tail_count_5s / 5.0 if tail_count_5s > 0 else 0
                
                # 调试输出，帮助确认数据来源
                print(f"🐟 Fish {fish_id}: 实时速度={realtime_speed_pixel:.2f}px/s, 真实速度={realtime_speed_real:.2f}mm/s, 尾摆频率={realtime_tail_frequency:.2f}Hz")
                
                data['fish_info'][fish_id] = {
                    'position': latest_pos,
                    'speed': realtime_speed_pixel,
                    'speed_real': realtime_speed_real,
                    'tail_frequency': realtime_tail_frequency,
                    'posture': fish_info.get('current_posture', 'unknown'),
                    'track_length': len(fish_info['positions'])
                }
                
        return data


class CalibrationDialog(QDialog):
    """相机标定对话框"""
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("相机标定")
        self.setModal(True)
        self.resize(600, 400)
        self.setup_ui()
        
    def setup_ui(self):
        layout = QVBoxLayout()
        
        # 标定参数设置
        params_group = QGroupBox("标定参数")
        params_layout = QGridLayout()
        
        params_layout.addWidget(QLabel("棋盘格列数:"), 0, 0)
        self.cols_spin = QSpinBox()
        self.cols_spin.setRange(3, 20)
        self.cols_spin.setValue(9)
        params_layout.addWidget(self.cols_spin, 0, 1)
        
        params_layout.addWidget(QLabel("棋盘格行数:"), 1, 0)
        self.rows_spin = QSpinBox()
        self.rows_spin.setRange(3, 20)
        self.rows_spin.setValue(6)
        params_layout.addWidget(self.rows_spin, 1, 1)
        
        params_layout.addWidget(QLabel("方格尺寸(mm):"), 2, 0)
        self.square_size_spin = QDoubleSpinBox()
        self.square_size_spin.setRange(1.0, 100.0)
        self.square_size_spin.setValue(25.0)
        params_layout.addWidget(self.square_size_spin, 2, 1)
        
        params_group.setLayout(params_layout)
        layout.addWidget(params_group)
        
        # 图像选择
        images_group = QGroupBox("标定图像")
        images_layout = QVBoxLayout()
        
        self.images_path_edit = QLineEdit()
        self.images_path_edit.setPlaceholderText("选择包含标定图像的文件夹...")
        
        browse_btn = QPushButton("浏览文件夹")
        browse_btn.clicked.connect(self.browse_images_folder)
        
        images_h_layout = QHBoxLayout()
        images_h_layout.addWidget(self.images_path_edit)
        images_h_layout.addWidget(browse_btn)
        images_layout.addLayout(images_h_layout)
        
        self.images_info_label = QLabel("未选择图像文件夹")
        images_layout.addWidget(self.images_info_label)
        
        images_group.setLayout(images_layout)
        layout.addWidget(images_group)
        
        # 进度和日志
        self.progress_bar = QProgressBar()
        layout.addWidget(self.progress_bar)
        
        self.log_text = QTextEdit()
        self.log_text.setMaximumHeight(150)
        layout.addWidget(self.log_text)
        
        # 按钮
        button_box = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        button_box.accepted.connect(self.start_calibration)
        button_box.rejected.connect(self.reject)
        layout.addWidget(button_box)
        
        self.setLayout(layout)
        
    def browse_images_folder(self):
        """浏览图像文件夹"""
        folder = QFileDialog.getExistingDirectory(self, "选择标定图像文件夹")
        if folder:
            self.images_path_edit.setText(folder)
            self.update_images_info(folder)
            
    def update_images_info(self, folder):
        """更新图像信息"""
        import glob
        images = glob.glob(os.path.join(folder, "*.jpg")) + \
                glob.glob(os.path.join(folder, "*.png"))
        self.images_info_label.setText(f"找到 {len(images)} 张图像")
        
    def start_calibration(self):
        """开始标定"""
        images_path = self.images_path_edit.text()
        if not images_path:
            QMessageBox.warning(self, "警告", "请选择标定图像文件夹")
            return
            
        try:
            calibrator = CameraCalibrator()
            
            self.log_text.append("开始相机标定...")
            self.progress_bar.setValue(20)
            
            checkerboard_size = (self.cols_spin.value(), self.rows_spin.value())
            square_size = self.square_size_spin.value()
            
            success = calibrator.calibrate_camera(
                images_path, checkerboard_size, square_size
            )
            
            self.progress_bar.setValue(80)
            
            if success:
                # 保存标定结果
                calibration_file = "camera_calibration.json"
                calibrator.save_calibration(calibration_file)
                
                self.log_text.append(f"标定成功！结果已保存到: {calibration_file}")
                self.progress_bar.setValue(100)
                
                QMessageBox.information(self, "成功", "相机标定完成！")
                self.accept()
            else:
                self.log_text.append("标定失败，请检查图像质量和参数设置")
                QMessageBox.warning(self, "失败", "相机标定失败")
                
        except Exception as e:
            self.log_text.append(f"标定过程出错: {str(e)}")
            QMessageBox.critical(self, "错误", f"标定过程出错: {str(e)}")


class HeatmapWidget(QWidget):
    """热力图显示组件"""
    def __init__(self):
        super().__init__()
        self.setMinimumSize(300, 200)
        self.positions = []
        self.setup_ui()
        
    def setup_ui(self):
        layout = QVBoxLayout()
        
        # 创建matplotlib图形
        self.figure = Figure(figsize=(6, 4))
        self.canvas = FigureCanvas(self.figure)
        layout.addWidget(self.canvas)
        
        self.setLayout(layout)
        
    def update_heatmap(self, fish_data):
        """更新热力图"""
        self.figure.clear()
        
        if not fish_data:
            self.canvas.draw()
            return
            
        # 收集所有位置数据
        all_positions = []
        for fish_info in fish_data.values():
            if 'position' in fish_info:
                pos = fish_info['position']
                if 'pixel' in pos:
                    all_positions.append([pos['pixel']['x'], pos['pixel']['y']])
                    
        if not all_positions:
            self.canvas.draw()
            return
            
        # 创建热力图
        ax = self.figure.add_subplot(111)
        positions = np.array(all_positions)
        
        # 使用hexbin创建六边形热力图
        hb = ax.hexbin(positions[:, 0], positions[:, 1], gridsize=20, cmap='YlOrRd')
        
        ax.set_title('Fish Distribution Heatmap')
        ax.set_xlabel('X Coordinate')
        ax.set_ylabel('Y Coordinate')
        
        # 添加颜色条
        cb = self.figure.colorbar(hb, ax=ax)
        cb.set_label('Density')
        
        self.canvas.draw()


class TrackingInfoWidget(QWidget):
    """跟踪信息显示组件"""
    def __init__(self):
        super().__init__()
        self.setup_ui()
        
    def setup_ui(self):
        layout = QVBoxLayout()
        
        # 全局统计
        stats_group = QGroupBox("全局统计")
        stats_layout = QGridLayout()
        
        self.fish_count_label = QLabel("鱼类数量: 0")
        self.active_count_label = QLabel("活跃数量: 0")
        self.avg_speed_label = QLabel("平均速度: 0.0")
        self.total_distance_label = QLabel("总距离: 0.0")
        
        stats_layout.addWidget(self.fish_count_label, 0, 0)
        stats_layout.addWidget(self.active_count_label, 0, 1)
        stats_layout.addWidget(self.avg_speed_label, 1, 0)
        stats_layout.addWidget(self.total_distance_label, 1, 1)
        
        stats_group.setLayout(stats_layout)
        layout.addWidget(stats_group)
        
        # 个体信息表格
        fish_group = QGroupBox("个体信息")
        fish_layout = QVBoxLayout()
        
        self.fish_table = QTableWidget()
        self.fish_table.setColumnCount(6)
        self.fish_table.setHorizontalHeaderLabels([
            "ID", "位置(X,Y)", "瞬时速度(px/s)", "瞬时频率(Hz)", "姿态", "追踪时长(s)"
        ])
        
        fish_layout.addWidget(self.fish_table)
        fish_group.setLayout(fish_layout)
        layout.addWidget(fish_group)
        
        self.setLayout(layout)
        
    def update_info(self, tracking_data):
        """更新跟踪信息"""
        if not tracking_data:
            return
            
        # 更新全局统计
        fish_count = tracking_data.get('fish_count', 0)
        total_fish_count = tracking_data.get('total_fish_count', 0)
        self.fish_count_label.setText(f"鱼类数量: {fish_count}/{total_fish_count}")
        
        global_stats = tracking_data.get('global_stats', {})
        self.avg_speed_label.setText(f"平均速度: {global_stats.get('avg_speed', 0):.2f}")
        self.total_distance_label.setText(f"总距离: {global_stats.get('total_distance', 0):.2f}")
        
        # 更新个体信息表格 - 只显示当前活跃的鱼
        fish_info = tracking_data.get('fish_info', {})
        # 过滤掉消失的鱼 - 只显示当前帧有位置信息的鱼
        active_fish_info = {fish_id: info for fish_id, info in fish_info.items() 
                           if info.get('position') is not None}
        
        self.fish_table.setRowCount(len(active_fish_info))
        
        for row, (fish_id, info) in enumerate(active_fish_info.items()):
            self.fish_table.setItem(row, 0, QTableWidgetItem(str(fish_id)))
            
            pos = info.get('position', {})
            if 'pixel' in pos:
                pos_text = f"({pos['pixel']['x']:.0f}, {pos['pixel']['y']:.0f})"
            else:
                pos_text = "N/A"
            self.fish_table.setItem(row, 1, QTableWidgetItem(pos_text))
            
            # 显示瞬时速度，保留2位小数
            speed = info.get('speed', 0)
            speed_text = f"{speed:.2f}" if speed > 0 else "0.00"
            self.fish_table.setItem(row, 2, QTableWidgetItem(speed_text))
            
            # 显示瞬时尾摆频率，保留2位小数
            freq = info.get('tail_frequency', 0)
            freq_text = f"{freq:.2f}" if freq > 0 else "0.00"
            self.fish_table.setItem(row, 3, QTableWidgetItem(freq_text))
            
            posture = info.get('posture', 'unknown')
            self.fish_table.setItem(row, 4, QTableWidgetItem(posture))
            
            # 追踪时长以秒为单位显示
            track_duration = info.get('track_duration', 0)
            track_duration_text = f"{track_duration:.1f}"
            self.fish_table.setItem(row, 5, QTableWidgetItem(track_duration_text))


class MainWindow(QMainWindow):
    """主窗口"""
    def __init__(self):
        super().__init__()
        self.setWindowTitle("鱼类跟踪系统 - 可视化界面")
        self.setGeometry(100, 100, 1400, 900)
        
        # 设置应用图标和样式
        self.setStyleSheet("""
            QMainWindow {
                background-color: #f0f0f0;
            }
            QGroupBox {
                font-weight: bold;
                border: 2px solid #cccccc;
                border-radius: 5px;
                margin-top: 1ex;
                padding-top: 10px;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 10px;
                padding: 0 5px 0 5px;
            }
            QPushButton {
                background-color: #4CAF50;
                color: white;
                border: none;
                padding: 8px 16px;
                border-radius: 4px;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #45a049;
            }
            QPushButton:pressed {
                background-color: #3d8b40;
            }
            QPushButton:disabled {
                background-color: #cccccc;
                color: #666666;
            }
        """)
        
        self.video_thread = VideoThread()
        self.video_thread.frame_ready.connect(self.update_video_display)
        self.video_thread.tracking_data_ready.connect(self.update_tracking_info)
        self.video_thread.error_occurred.connect(self.show_error)
        
        self.setup_ui()
        self.is_tracking = False
        
    def setup_ui(self):
        """设置用户界面"""
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        
        # 主布局
        main_layout = QHBoxLayout()
        
        # 左侧控制面板
        control_panel = self.create_control_panel()
        main_layout.addWidget(control_panel, 1)
        
        # 右侧显示区域
        display_area = self.create_display_area()
        main_layout.addWidget(display_area, 3)
        
        central_widget.setLayout(main_layout)
        
    def create_control_panel(self):
        """创建控制面板"""
        panel = QWidget()
        panel.setMaximumWidth(400)
        layout = QVBoxLayout()
        
        # 模型选择
        model_group = QGroupBox("模型设置")
        model_layout = QGridLayout()
        
        model_layout.addWidget(QLabel("模型文件:"), 0, 0)
        self.model_combo = QComboBox()
        self.update_model_list()
        model_layout.addWidget(self.model_combo, 0, 1)
        
        browse_model_btn = QPushButton("浏览")
        browse_model_btn.clicked.connect(self.browse_model_file)
        model_layout.addWidget(browse_model_btn, 0, 2)
        
        model_layout.addWidget(QLabel("跟踪器:"), 1, 0)
        self.tracker_combo = QComboBox()
        self.tracker_combo.addItems(["bytetrack.yaml", "botsort.yaml"])
        model_layout.addWidget(self.tracker_combo, 1, 1, 1, 2)
        
        model_layout.addWidget(QLabel("置信度:"), 2, 0)
        self.conf_slider = QSlider(Qt.Horizontal)
        self.conf_slider.setRange(1, 100)
        self.conf_slider.setValue(50)
        self.conf_slider.valueChanged.connect(self.update_conf_label)
        model_layout.addWidget(self.conf_slider, 2, 1)
        
        self.conf_label = QLabel("0.50")
        model_layout.addWidget(self.conf_label, 2, 2)
        
        model_group.setLayout(model_layout)
        layout.addWidget(model_group)
        
        # 视频源选择
        video_group = QGroupBox("视频源")
        video_layout = QGridLayout()
        
        self.video_file_radio = QCheckBox("视频文件")
        self.video_file_radio.setChecked(True)
        video_layout.addWidget(self.video_file_radio, 0, 0)
        
        self.camera_radio = QCheckBox("摄像头")
        video_layout.addWidget(self.camera_radio, 0, 1)
        
        self.video_path_edit = QLineEdit()
        self.video_path_edit.setPlaceholderText("选择视频文件...")
        video_layout.addWidget(self.video_path_edit, 1, 0, 1, 2)
        
        browse_video_btn = QPushButton("浏览视频")
        browse_video_btn.clicked.connect(self.browse_video_file)
        video_layout.addWidget(browse_video_btn, 2, 0)
        
        self.camera_combo = QComboBox()
        self.camera_combo.addItems(["摄像头 0", "摄像头 1", "摄像头 2"])
        self.camera_combo.setEnabled(False)
        video_layout.addWidget(self.camera_combo, 2, 1)
        
        # 连接单选框事件
        self.video_file_radio.toggled.connect(self.update_video_source_controls)
        self.camera_radio.toggled.connect(self.update_video_source_controls)
        
        video_group.setLayout(video_layout)
        layout.addWidget(video_group)
        
        # 畸变矫正设置
        correction_group = QGroupBox("畸变矫正")
        correction_layout = QGridLayout()
        
        self.enable_correction_check = QCheckBox("启用畸变矫正")
        correction_layout.addWidget(self.enable_correction_check, 0, 0, 1, 2)
        
        correction_layout.addWidget(QLabel("标定文件:"), 1, 0)
        self.calibration_combo = QComboBox()
        self.update_calibration_list()
        correction_layout.addWidget(self.calibration_combo, 1, 1)
        
        calibrate_btn = QPushButton("相机标定")
        calibrate_btn.clicked.connect(self.open_calibration_dialog)
        correction_layout.addWidget(calibrate_btn, 2, 0, 1, 2)
        
        correction_group.setLayout(correction_layout)
        layout.addWidget(correction_group)
        
        # 处理模式设置
        mode_group = QGroupBox("处理模式")
        mode_layout = QGridLayout()
        
        # 创建按钮组确保互斥
        self.mode_button_group = QButtonGroup()
        
        self.realtime_radio = QRadioButton("实时模式")
        self.realtime_radio.setChecked(True)
        mode_layout.addWidget(self.realtime_radio, 0, 0)
        self.mode_button_group.addButton(self.realtime_radio)
        
        self.batch_radio = QRadioButton("批处理模式")
        mode_layout.addWidget(self.batch_radio, 0, 1)
        self.mode_button_group.addButton(self.batch_radio)
        
        # 连接模式切换事件
        self.realtime_radio.toggled.connect(self.update_mode_controls)
        self.batch_radio.toggled.connect(self.update_mode_controls)
        
        mode_group.setLayout(mode_layout)
        layout.addWidget(mode_group)
        
        # 性能配置设置
        perf_group = QGroupBox("性能配置")
        perf_layout = QGridLayout()
        
        # CUDA选项
        self.cuda_checkbox = QCheckBox("使用CUDA加速")
        self.cuda_checkbox.setChecked(True)  # 默认启用
        perf_layout.addWidget(self.cuda_checkbox, 0, 0, 1, 2)
        
        # 批处理保存间隔
        perf_layout.addWidget(QLabel("批处理保存间隔:"), 1, 0)
        self.save_interval_spin = QSpinBox()
        self.save_interval_spin.setRange(1, 10)
        self.save_interval_spin.setValue(1)
        self.save_interval_spin.setSuffix("秒")
        perf_layout.addWidget(self.save_interval_spin, 1, 1)
        
        # 性能提示
        perf_tip = QLabel("提示: CUDA可显著提升处理速度")
        perf_tip.setStyleSheet("color: #666; font-size: 10px;")
        perf_layout.addWidget(perf_tip, 2, 0, 1, 2)
        
        perf_group.setLayout(perf_layout)
        layout.addWidget(perf_group)
        
        # 数据过滤设置
        filter_group = QGroupBox("数据过滤")
        filter_layout = QGridLayout()
        
        self.filter_enabled_check = QCheckBox("启用过滤")
        self.filter_enabled_check.setChecked(True)
        filter_layout.addWidget(self.filter_enabled_check, 0, 0, 1, 2)
        
        filter_layout.addWidget(QLabel("最小追踪时长:"), 1, 0)
        self.min_track_length_spin = QSpinBox()
        self.min_track_length_spin.setRange(1, 1000)
        self.min_track_length_spin.setValue(10)
        self.min_track_length_spin.setSuffix(" 帧")
        filter_layout.addWidget(self.min_track_length_spin, 1, 1)
        
        filter_layout.addWidget(QLabel("最小尾摆频率:"), 2, 0)
        self.min_freq_spin = QDoubleSpinBox()
        self.min_freq_spin.setRange(0.0, 10.0)
        self.min_freq_spin.setSingleStep(0.1)
        self.min_freq_spin.setValue(0.1)
        filter_layout.addWidget(self.min_freq_spin, 2, 1)
        
        filter_layout.addWidget(QLabel("最小平均速度:"), 3, 0)
        self.min_speed_spin = QDoubleSpinBox()
        self.min_speed_spin.setRange(0.0, 100.0)
        self.min_speed_spin.setValue(1.0)
        filter_layout.addWidget(self.min_speed_spin, 3, 1)
        
        filter_group.setLayout(filter_layout)
        layout.addWidget(filter_group)
        
        # 控制按钮
        control_group = QGroupBox("控制")
        control_layout = QVBoxLayout()
        
        self.start_btn = QPushButton("开始跟踪")
        self.start_btn.clicked.connect(self.toggle_tracking)
        control_layout.addWidget(self.start_btn)
        
        self.save_btn = QPushButton("保存数据")
        self.save_btn.clicked.connect(self.save_tracking_data)
        self.save_btn.setEnabled(False)
        control_layout.addWidget(self.save_btn)
        
        self.report_btn = QPushButton("生成报告")
        self.report_btn.clicked.connect(self.generate_report)
        self.report_btn.setEnabled(False)
        control_layout.addWidget(self.report_btn)
        
        control_group.setLayout(control_layout)
        layout.addWidget(control_group)
        
        # 添加弹性空间
        layout.addStretch()
        
        panel.setLayout(layout)
        return panel
        
    def create_display_area(self):
        """创建显示区域"""
        # 使用分割器
        splitter = QSplitter(Qt.Vertical)
        
        # 上半部分：视频显示
        video_widget = QWidget()
        video_layout = QVBoxLayout()
        
        self.video_label = QLabel("视频显示区域")
        self.video_label.setMinimumSize(640, 480)
        self.video_label.setStyleSheet("""
            QLabel {
                border: 2px solid #cccccc;
                background-color: black;
                color: white;
                text-align: center;
                font-size: 16px;
            }
        """)
        self.video_label.setAlignment(Qt.AlignCenter)
        video_layout.addWidget(self.video_label)
        
        video_widget.setLayout(video_layout)
        splitter.addWidget(video_widget)
        
        # 下半部分：信息面板（使用标签页）
        self.info_tabs = QTabWidget()
        
        # 跟踪信息标签页
        self.tracking_info_widget = TrackingInfoWidget()
        self.info_tabs.addTab(self.tracking_info_widget, "跟踪信息")
        
        # 热力图标签页
        self.heatmap_widget = HeatmapWidget()
        self.info_tabs.addTab(self.heatmap_widget, "分布热力图")
        
        # 日志标签页
        self.log_text = QTextEdit()
        self.log_text.setMaximumHeight(200)
        self.info_tabs.addTab(self.log_text, "日志")
        
        splitter.addWidget(self.info_tabs)
        
        # 设置分割器比例
        splitter.setSizes([600, 300])
        
        return splitter
        
    def update_model_list(self):
        """更新模型列表"""
        models = []
        for file in os.listdir('.'):
            if file.endswith(('.pt', '.pth')):
                models.append(file)
        
        self.model_combo.clear()
        if models:
            self.model_combo.addItems(models)
            # 设置默认模型
            if 'bestx.pt' in models:
                self.model_combo.setCurrentText('bestx.pt')
        else:
            self.model_combo.addItem("未找到模型文件")
            
    def update_calibration_list(self):
        """更新标定文件列表"""
        calibrations = []
        for file in os.listdir('.'):
            if file.endswith(('.json', '.npz')):
                calibrations.append(file)
                
        self.calibration_combo.clear()
        if calibrations:
            self.calibration_combo.addItems(calibrations)
            # 设置默认标定文件
            if 'camera_calibration.json' in calibrations:
                self.calibration_combo.setCurrentText('camera_calibration.json')
        else:
            self.calibration_combo.addItem("未找到标定文件")
            
    def update_conf_label(self, value):
        """更新置信度标签"""
        self.conf_label.setText(f"{value/100:.2f}")
        
    def update_video_source_controls(self):
        """更新视频源控件状态"""
        if self.video_file_radio.isChecked():
            self.video_path_edit.setEnabled(True)
            self.camera_combo.setEnabled(False)
        else:
            self.video_path_edit.setEnabled(False)
            self.camera_combo.setEnabled(True)
    
    def update_mode_controls(self):
        """更新处理模式控件状态"""
        # RadioButton自动互斥，不需要额外处理
        if self.batch_radio.isChecked():
            self.log_text.append("切换到批处理模式")
        else:
            self.log_text.append("切换到实时模式")
            
    def browse_model_file(self):
        """浏览模型文件"""
        file_path, _ = QFileDialog.getOpenFileName(
            self, "选择模型文件", "", "模型文件 (*.pt *.pth);;所有文件 (*)"
        )
        if file_path:
            model_name = os.path.basename(file_path)
            if model_name not in [self.model_combo.itemText(i) for i in range(self.model_combo.count())]:
                self.model_combo.addItem(model_name)
            self.model_combo.setCurrentText(model_name)
            
    def browse_video_file(self):
        """浏览视频文件"""
        file_path, _ = QFileDialog.getOpenFileName(
            self, "选择视频文件", "", "视频文件 (*.mp4 *.avi *.mov *.mkv);;所有文件 (*)"
        )
        if file_path:
            self.video_path_edit.setText(file_path)
            
    def open_calibration_dialog(self):
        """打开标定对话框"""
        dialog = CalibrationDialog(self)
        if dialog.exec_() == QDialog.Accepted:
            self.update_calibration_list()
            self.log_text.append("相机标定完成")
            
    def toggle_tracking(self):
        """切换跟踪状态"""
        if not self.is_tracking:
            self.start_tracking()
        else:
            self.stop_tracking()
            
    def start_tracking(self):
        """开始跟踪"""
        try:
            # 获取设置
            model_path = self.model_combo.currentText()
            if model_path == "未找到模型文件":
                QMessageBox.warning(self, "警告", "请选择有效的模型文件")
                return
                
            tracker_type = self.tracker_combo.currentText()
            conf_threshold = self.conf_slider.value() / 100.0
            enable_correction = self.enable_correction_check.isChecked()
            calibration_file = self.calibration_combo.currentText() if enable_correction else None
            batch_mode = self.batch_radio.isChecked()
            
            if enable_correction and calibration_file == "未找到标定文件":
                QMessageBox.warning(self, "警告", "启用畸变矫正需要有效的标定文件")
                return
                
            # 获取视频源
            if self.video_file_radio.isChecked():
                video_source = self.video_path_edit.text()
                if not video_source:
                    QMessageBox.warning(self, "警告", "请选择视频文件")
                    return
            else:
                if batch_mode:
                    QMessageBox.warning(self, "警告", "批处理模式不支持摄像头输入，请选择视频文件")
                    return
                camera_index = self.camera_combo.currentIndex()
                video_source = camera_index
            
            # 批处理模式特殊处理
            if batch_mode:
                use_cuda = self.cuda_checkbox.isChecked()
                save_interval = self.save_interval_spin.value()
                self.start_batch_processing(model_path, tracker_type, conf_threshold, 
                                          enable_correction, calibration_file, video_source,
                                          use_cuda, save_interval)
                return
                
            # 获取性能设置
            use_cuda = self.cuda_checkbox.isChecked()
            save_interval = self.save_interval_spin.value()
            
            # 设置跟踪器
            self.video_thread.setup_tracker(
                model_path, tracker_type, conf_threshold,
                enable_correction, calibration_file, batch_mode,
                use_cuda, save_interval
            )
            
            # 设置视频源并开始
            self.video_thread.set_video_source(video_source)
            self.video_thread.start()
            
            self.is_tracking = True
            self.start_btn.setText("停止跟踪")
            self.save_btn.setEnabled(True)
            self.report_btn.setEnabled(True)
            
            self.log_text.append(f"开始跟踪 - 模型: {model_path}, 视频源: {video_source}")
            
        except Exception as e:
            QMessageBox.critical(self, "错误", f"启动跟踪失败: {str(e)}")
    
    def start_batch_processing(self, model_path, tracker_type, conf_threshold, 
                              enable_correction, calibration_file, video_source,
                              use_cuda=True, save_interval=1.0):
        """开始批处理"""
        try:
            from fish_tracking_advanced import FishTracker
            import cv2
            
            # 创建批处理跟踪器（支持CUDA和保存间隔）
            tracker = FishTracker(
                model_path=model_path,
                tracker=tracker_type,
                conf=conf_threshold,
                enable_distortion_correction=enable_correction,
                calibration_file=calibration_file,
                batch_mode=True,
                output_dir="batch_output",
                use_cuda=use_cuda,
                save_interval=save_interval
            )
            
            self.log_text.append(f"批处理设置 - CUDA: {use_cuda}, 保存间隔: {save_interval}秒")
            
            # 处理视频
            cap = cv2.VideoCapture(video_source)
            
            if not cap.isOpened():
                QMessageBox.critical(self, "错误", "无法打开视频文件")
                return
            
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            
            # 创建进度对话框
            progress = QProgressDialog("正在批处理视频...", "取消", 0, total_frames, self)
            progress.setWindowModality(Qt.WindowModal)
            progress.show()
            
            frame_count = 0
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                # 处理帧
                tracker.process_frame(frame)
                
                frame_count += 1
                progress.setValue(frame_count)
                
                if progress.wasCanceled():
                    break
                
                # 更新GUI
                QApplication.processEvents()
            
            cap.release()
            progress.close()
            
            if not progress.wasCanceled():
                # 保存批处理结果
                tracker.save_batch_summary()
                
                # 获取过滤参数并安全调用
                try:
                    filter_params = {
                        'min_track_length': self.min_track_length_spin.value(),
                        'min_tail_frequency': self.min_freq_spin.value(),
                        'min_speed': self.min_speed_spin.value()
                    }
                    
                    valid_count = len(tracker.filter_valid_fish_ids(**filter_params))
                except Exception as filter_error:
                    print(f"过滤错误: {filter_error}")
                    valid_count = len(tracker.fish_data) if hasattr(tracker, 'fish_data') else 0
                
                QMessageBox.information(self, "完成", 
                                      f"批处理完成！\n"
                                      f"处理帧数: {frame_count}\n"
                                      f"输出目录: batch_output\n"
                                      f"总鱼类ID: {len(tracker.fish_data)}\n"
                                      f"有效鱼类ID: {valid_count}")
                
                self.log_text.append(f"批处理完成 - 总帧数: {frame_count}, 有效鱼类: {valid_count}")
            else:
                self.log_text.append("批处理被用户取消")
            
        except Exception as e:
            import traceback
            error_details = traceback.format_exc()
            print(f"批处理详细错误: {error_details}")
            QMessageBox.critical(self, "错误", f"批处理失败: {str(e)}\n\n查看控制台获取详细错误信息")
            self.log_text.append(f"批处理错误: {str(e)}")
            self.log_text.append("详细错误信息已打印到控制台")
            
    def stop_tracking(self):
        """停止跟踪"""
        self.video_thread.stop()
        self.is_tracking = False
        self.start_btn.setText("开始跟踪")
        
        # 保存处理后的视频
        if hasattr(self.video_thread, 'processed_frames') and self.video_thread.processed_frames:
            self._save_processed_video()
        
        self.log_text.append("跟踪已停止")
        
    def update_video_display(self, frame):
        """更新视频显示"""
        try:
            # 转换为Qt图像格式
            rgb_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            h, w, ch = rgb_image.shape
            bytes_per_line = ch * w
            qt_image = QImage(rgb_image.data, w, h, bytes_per_line, QImage.Format_RGB888)
            
            # 缩放图像以适应标签
            label_size = self.video_label.size()
            scaled_pixmap = QPixmap.fromImage(qt_image).scaled(
                label_size, Qt.KeepAspectRatio, Qt.SmoothTransformation
            )
            
            self.video_label.setPixmap(scaled_pixmap)
            
        except Exception as e:
            self.log_text.append(f"视频显示错误: {str(e)}")
            
    def update_tracking_info(self, tracking_data):
        """更新跟踪信息"""
        try:
            # 更新跟踪信息表格
            self.tracking_info_widget.update_info(tracking_data)
            
            # 更新热力图
            fish_info = tracking_data.get('fish_info', {})
            self.heatmap_widget.update_heatmap(fish_info)
            
        except Exception as e:
            self.log_text.append(f"信息更新错误: {str(e)}")
            
    def show_error(self, error_message):
        """显示错误信息"""
        self.log_text.append(f"错误: {error_message}")
        QMessageBox.critical(self, "错误", error_message)
        
    def save_tracking_data(self):
        """保存跟踪数据"""
        if not self.video_thread.tracker:
            QMessageBox.warning(self, "警告", "没有可保存的跟踪数据")
            return
            
        file_path, _ = QFileDialog.getSaveFileName(
            self, "保存跟踪数据", f"tracking_data_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
            "CSV文件 (*.csv);;所有文件 (*)"
        )
        
        if file_path:
            try:
                import csv
                tracker = self.video_thread.tracker
                
                # 获取过滤参数
                filter_params = {
                    'min_track_length': self.min_track_length_spin.value(),
                    'min_tail_frequency': self.min_freq_spin.value(),
                    'min_speed': self.min_speed_spin.value()
                }
                
                # 获取有效的鱼类数据
                if hasattr(tracker, 'filter_valid_fish_ids'):
                    valid_fish_ids = tracker.filter_valid_fish_ids(**filter_params)
                    fish_data = {fid: tracker.fish_data[fid] for fid in valid_fish_ids if fid in tracker.fish_data}
                else:
                    fish_data = tracker.fish_data
                
                # 写入CSV文件 - 改为按帧记录
                with open(file_path, 'w', newline='', encoding='utf-8') as csvfile:
                    fieldnames = [
                        '鱼类ID', '帧号', '时间戳(秒)', '鱼尾位置X(像素)', '鱼尾位置Y(像素)', 
                        '鱼尾位置X(毫米)', '鱼尾位置Y(毫米)', '尾部角度(度)', '尾摆频率(次/5秒)', 
                        '游泳速度(像素/秒)', '游泳速度(毫米/秒)', '鱼体长度(像素)', '鱼体长度(毫米)', 
                        '体态', '畸变矫正', '真实尺寸可用'
                    ]
                    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                    writer.writeheader()
                    
                    # 从tracking_data获取按帧记录的数据
                    if hasattr(tracker, 'tracking_data'):
                        print(f"🗃️ CSV导出：使用tracking_data，共{len(tracker.tracking_data)}条鱼的数据")
                        for fish_id, tracking_entries in tracker.tracking_data.items():
                            print(f"🐟 鱼类{fish_id}: 共{len(tracking_entries)}帧数据")
                            for entry in tracking_entries:
                                # 获取尾部位置
                                tail_pos_pixel = entry.get('tail_position_pixel', (0, 0))
                                tail_pos_real = entry.get('tail_position_real', (0, 0)) if entry.get('real_scale_available', False) else (0, 0)
                                
                                row = {
                                    '鱼类ID': fish_id,
                                    '帧号': entry.get('frame', 0),
                                    '时间戳(秒)': entry.get('timestamp', 0),
                                    '鱼尾位置X(像素)': tail_pos_pixel[0] if tail_pos_pixel else 0,
                                    '鱼尾位置Y(像素)': tail_pos_pixel[1] if tail_pos_pixel else 0,
                                    '鱼尾位置X(毫米)': tail_pos_real[0] if tail_pos_real and entry.get('real_scale_available', False) else 0,
                                    '鱼尾位置Y(毫米)': tail_pos_real[1] if tail_pos_real and entry.get('real_scale_available', False) else 0,
                                    '尾部角度(度)': entry.get('tail_angle', 0),
                                    '尾摆频率(次/5秒)': entry.get('tail_freq', 0),
                                    '游泳速度(像素/秒)': entry.get('speed_pixel', 0),
                                    '游泳速度(毫米/秒)': entry.get('speed_real', 0) if entry.get('real_scale_available', False) else 0,
                                    '鱼体长度(像素)': entry.get('fish_length_pixel', 0),
                                    '鱼体长度(毫米)': entry.get('fish_length_real', 0) if entry.get('real_scale_available', False) else 0,
                                    '体态': entry.get('posture', 'Unknown'),
                                    '畸变矫正': '是' if entry.get('corrected_available', False) else '否',
                                    '真实尺寸可用': '是' if entry.get('real_scale_available', False) else '否'
                                }
                                
                                # 调试输出：检查关键数据
                                speed = row['游泳速度(像素/秒)']
                                freq = row['尾摆频率(次/5秒)']
                                if speed == 0 or freq == 0:
                                    print(f"⚠️ 鱼类{fish_id}帧{row['帧号']}: 速度={speed}, 频率={freq}, entry中原始数据: speed_pixel={entry.get('speed_pixel', 'missing')}, tail_freq={entry.get('tail_freq', 'missing')}")
                                
                                writer.writerow(row)
                    else:
                        # 兼容旧版本 - 如果没有tracking_data，使用fish_data
                        for fish_id, fish_info in fish_data.items():
                            if fish_info.get('positions'):
                                for frame_idx, pos in enumerate(fish_info['positions']):
                                    row = {
                                        '鱼类ID': fish_id,
                                        '帧号': frame_idx,
                                        '时间戳(秒)': frame_idx / 30.0,  # 假设30fps
                                        '鱼尾位置X(像素)': pos[0] if isinstance(pos, (list, tuple)) else pos.get('x', 0),
                                        '鱼尾位置Y(像素)': pos[1] if isinstance(pos, (list, tuple)) else pos.get('y', 0),
                                        '鱼尾位置X(毫米)': 0,
                                        '鱼尾位置Y(毫米)': 0,
                                        '尾部角度(度)': 0,
                                        '尾摆频率(次/5秒)': fish_info.get('tail_frequency', 0),
                                        '游泳速度(像素/秒)': fish_info.get('speeds', [0])[frame_idx] if frame_idx < len(fish_info.get('speeds', [])) else 0,
                                        '游泳速度(毫米/秒)': 0,
                                        '鱼体长度(像素)': 0,
                                        '鱼体长度(毫米)': 0,
                                        '体态': fish_info.get('current_posture', 'Unknown'),
                                        '畸变矫正': '否',
                                        '真实尺寸可用': '否'
                                    }
                                    writer.writerow(row)
                
                # 保存全局统计信息
                stats_file = file_path.replace('.csv', '_stats.json')
                import json
                with open(stats_file, 'w', encoding='utf-8') as f:
                    json.dump({
                        'total_fish_count': len(tracker.fish_data),
                        'valid_fish_count': len(fish_data),
                        'global_stats': tracker.global_stats if hasattr(tracker, 'global_stats') else {},
                        'filter_params': filter_params,
                        'export_time': datetime.now().isoformat()
                    }, f, ensure_ascii=False, indent=2, default=str)
                
                self.log_text.append(f"跟踪数据已保存到: {file_path}")
                self.log_text.append(f"统计信息已保存到: {stats_file}")
                QMessageBox.information(self, "成功", f"跟踪数据保存成功！\n数据文件: {file_path}\n统计文件: {stats_file}")
                
            except Exception as e:
                import traceback
                error_details = traceback.format_exc()
                print(f"保存数据详细错误: {error_details}")
                QMessageBox.critical(self, "错误", f"保存失败: {str(e)}")
                self.log_text.append(f"保存错误: {str(e)}")
                
    def generate_report(self):
        """生成报告"""
        if not self.video_thread.tracker:
            QMessageBox.warning(self, "警告", "没有可用的跟踪数据")
            return
            
        try:
            self.log_text.append("正在生成报告...")
            tracker = self.video_thread.tracker
            
            # 获取过滤参数
            filter_params = {
                'min_track_length': self.min_track_length_spin.value(),
                'min_tail_frequency': self.min_freq_spin.value(),
                'min_speed': self.min_speed_spin.value()
            }
            
            # 获取有效的鱼类数据
            if hasattr(tracker, 'filter_valid_fish_ids'):
                valid_fish_ids = tracker.filter_valid_fish_ids(**filter_params)
                fish_data = {fid: tracker.fish_data[fid] for fid in valid_fish_ids if fid in tracker.fish_data}
            else:
                fish_data = tracker.fish_data
            
            # 创建报告目录
            report_dir = f"report_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            os.makedirs(report_dir, exist_ok=True)
            
            # 生成HTML报告
            html_content = self._generate_html_report(fish_data, tracker, filter_params)
            report_file = os.path.join(report_dir, "tracking_report.html")
            with open(report_file, 'w', encoding='utf-8') as f:
                f.write(html_content)
            
            # 生成统计图表
            if fish_data:
                self._generate_charts(fish_data, report_dir)
            
            self.log_text.append(f"报告已生成: {report_dir}")
            QMessageBox.information(self, "成功", f"报告生成完成！\n报告目录: {report_dir}")
            
        except Exception as e:
            import traceback
            error_details = traceback.format_exc()
            print(f"生成报告详细错误: {error_details}")
            QMessageBox.critical(self, "错误", f"报告生成失败: {str(e)}")
            self.log_text.append(f"报告生成错误: {str(e)}")
    
    def _generate_html_report(self, fish_data, tracker, filter_params):
        """生成HTML报告"""
        total_fish = len(tracker.fish_data) if hasattr(tracker, 'fish_data') else 0
        valid_fish = len(fish_data)
        
        # 计算统计信息 - 从tracking_data获取真实速度
        avg_speeds_real = []
        tail_freqs = []
        track_distances = []
        tracking_durations = []
        
        for fish_id, fish_info in fish_data.items():
            # 从tracking_data获取真实平均速度
            tracking_entries = []
            if hasattr(tracker, 'tracking_data') and fish_id in tracker.tracking_data:
                tracking_entries = tracker.tracking_data[fish_id]
            
            # 计算平均速度（优先使用真实速度）
            speeds_real = [entry.get('speed_real', 0) for entry in tracking_entries if entry.get('speed_real', 0) > 0]
            if speeds_real:
                avg_speeds_real.append(sum(speeds_real) / len(speeds_real))
            
            # 尾摆频率
            freqs = [entry.get('tail_freq', 0) for entry in tracking_entries]
            if freqs:
                tail_freqs.append(sum(freqs) / len(freqs))
            
            # 计算轨迹距离（游泳距离）
            if tracking_entries and len(tracking_entries) > 1:
                total_distance_pixel = 0
                total_distance_real = 0
                for i in range(1, len(tracking_entries)):
                    prev_entry = tracking_entries[i-1]
                    curr_entry = tracking_entries[i]
                    
                    # 像素距离
                    prev_pos = prev_entry.get('tail_position_pixel', (0, 0))
                    curr_pos = curr_entry.get('tail_position_pixel', (0, 0))
                    if prev_pos and curr_pos:
                        pixel_dist = ((curr_pos[0] - prev_pos[0])**2 + (curr_pos[1] - prev_pos[1])**2)**0.5
                        total_distance_pixel += pixel_dist
                    
                    # 真实距离
                    if curr_entry.get('real_scale_available', False):
                        prev_pos_real = prev_entry.get('tail_position_real', (0, 0))
                        curr_pos_real = curr_entry.get('tail_position_real', (0, 0))
                        if prev_pos_real and curr_pos_real:
                            real_dist = ((curr_pos_real[0] - prev_pos_real[0])**2 + (curr_pos_real[1] - prev_pos_real[1])**2)**0.5
                            total_distance_real += real_dist
                
                if total_distance_real > 0:
                    track_distances.append(total_distance_real)
                
                # 追踪时长
                start_time = tracking_entries[0].get('timestamp', 0)
                end_time = tracking_entries[-1].get('timestamp', 0)
                tracking_durations.append(end_time - start_time)
        
        avg_speed = sum(avg_speeds_real) / len(avg_speeds_real) if avg_speeds_real else 0
        avg_freq = sum(tail_freqs) / len(tail_freqs) if tail_freqs else 0
        avg_distance = sum(track_distances) / len(track_distances) if track_distances else 0
        avg_duration = sum(tracking_durations) / len(tracking_durations) if tracking_durations else 0
        
        html_content = f"""
<!DOCTYPE html>
<html>
<head>
    <meta charset="utf-8">
    <title>鱼类跟踪分析报告</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 20px; }}
        .header {{ background-color: #2196F3; color: white; padding: 20px; border-radius: 5px; }}
        .section {{ margin: 20px 0; padding: 15px; border: 1px solid #ddd; border-radius: 5px; }}
        .stats {{ display: flex; justify-content: space-around; }}
        .stat-box {{ text-align: center; padding: 10px; background-color: #f5f5f5; border-radius: 5px; }}
        .stat-number {{ font-size: 24px; font-weight: bold; color: #2196F3; }}
        table {{ width: 100%; border-collapse: collapse; margin-top: 10px; }}
        th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
        th {{ background-color: #f2f2f2; }}
    </style>
</head>
<body>
    <div class="header">
        <h1>🐟 鱼类跟踪分析报告</h1>
        <p>生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
    </div>
    
    <div class="section">
        <h2>📊 总体统计</h2>
        <div class="stats">
            <div class="stat-box">
                <div class="stat-number">{total_fish}</div>
                <div>检测到的鱼类总数</div>
            </div>
            <div class="stat-box">
                <div class="stat-number">{valid_fish}</div>
                <div>有效鱼类数量</div>
            </div>
            <div class="stat-box">
                <div class="stat-number">{avg_speed:.3f}</div>
                <div>平均游泳速度(毫米/秒)</div>
            </div>
            <div class="stat-box">
                <div class="stat-number">{avg_freq:.2f}</div>
                <div>平均尾摆频率(Hz)</div>
            </div>
            <div class="stat-box">
                <div class="stat-number">{avg_distance:.1f}</div>
                <div>平均轨迹距离(毫米)</div>
            </div>
            <div class="stat-box">
                <div class="stat-number">{avg_duration:.1f}</div>
                <div>平均追踪时长(秒)</div>
            </div>
        </div>
    </div>
    
    <div class="section">
        <h2>⚙️ 过滤参数</h2>
        <ul>
            <li>最小追踪时长: {filter_params['min_track_length']} 帧</li>
            <li>最小尾摆频率: {filter_params['min_tail_frequency']} Hz</li>
            <li>最小游泳速度: {filter_params['min_speed']}</li>
        </ul>
    </div>
    
    <div class="section">
        <h2>📋 鱼类列表与详细数据</h2>
        <table>
            <tr>
                <th>鱼类ID</th>
                <th>追踪时长(秒)</th>
                <th>轨迹距离(像素)</th>
                <th>轨迹距离(毫米)</th>
                <th>平均速度(毫米/秒)</th>
                <th>平均鱼体长度(毫米)</th>
                <th>平均尾摆频率(次/5秒)</th>
                <th>主要体态</th>
                <th>帧级数据</th>
            </tr>
        """
        
        # 从tracker获取tracking_data
        for fish_id, fish_info in fish_data.items():
            # 计算统计信息
            tracking_entries = []
            if hasattr(tracker, 'tracking_data') and fish_id in tracker.tracking_data:
                tracking_entries = tracker.tracking_data[fish_id]
            
            # 计算追踪时长
            tracking_duration = 0
            if tracking_entries and len(tracking_entries) > 1:
                start_time = tracking_entries[0].get('timestamp', 0)
                end_time = tracking_entries[-1].get('timestamp', 0)
                tracking_duration = end_time - start_time
            
            # 计算轨迹距离
            track_distance_pixel = 0
            track_distance_real = 0
            if tracking_entries and len(tracking_entries) > 1:
                for i in range(1, len(tracking_entries)):
                    prev_entry = tracking_entries[i-1]
                    curr_entry = tracking_entries[i]
                    
                    # 像素距离
                    prev_pos = prev_entry.get('tail_position_pixel', (0, 0))
                    curr_pos = curr_entry.get('tail_position_pixel', (0, 0))
                    if prev_pos and curr_pos:
                        pixel_dist = ((curr_pos[0] - prev_pos[0])**2 + (curr_pos[1] - prev_pos[1])**2)**0.5
                        track_distance_pixel += pixel_dist
                    
                    # 真实距离
                    if curr_entry.get('real_scale_available', False):
                        prev_pos_real = prev_entry.get('tail_position_real', (0, 0))
                        curr_pos_real = curr_entry.get('tail_position_real', (0, 0))
                        if prev_pos_real and curr_pos_real:
                            real_dist = ((curr_pos_real[0] - prev_pos_real[0])**2 + (curr_pos_real[1] - prev_pos_real[1])**2)**0.5
                            track_distance_real += real_dist
            
            # 计算平均速度（优先使用真实速度）
            speeds_real = [entry.get('speed_real', 0) for entry in tracking_entries if entry.get('speed_real', 0) > 0]
            avg_speed_real = sum(speeds_real) / len(speeds_real) if speeds_real else 0
            
            # 计算平均鱼体长度（毫米）
            fish_lengths_real = [entry.get('fish_length_real', 0) for entry in tracking_entries if entry.get('fish_length_real', 0) > 0]
            avg_fish_length_real = sum(fish_lengths_real) / len(fish_lengths_real) if fish_lengths_real else 0
            
            # 计算平均尾摆频率
            freqs = [entry.get('tail_freq', 0) for entry in tracking_entries]
            avg_freq = sum(freqs) / len(freqs) if freqs else 0
            
            # 主要体态（出现最多的）
            postures = [entry.get('posture', 'Unknown') for entry in tracking_entries]
            main_posture = max(set(postures), key=postures.count) if postures else 'Unknown'
            
            # 生成帧数据选择器（显示所有帧，不限制50帧）
            frame_data_html = ""
            if tracking_entries:
                frame_data_html = f'<select id="fish_{fish_id}_frames" onchange="showFrameData({fish_id})">'
                frame_data_html += '<option value="">选择帧号</option>'
                for i, entry in enumerate(tracking_entries):  # 显示所有帧
                    frame_num = entry.get('frame', i)
                    frame_data_html += f'<option value="{i}">第{frame_num}帧</option>'
                frame_data_html += '</select>'
                frame_data_html += f'<div id="fish_{fish_id}_detail" style="margin-top: 10px; display: none; background: #f9f9f9; padding: 10px; border-radius: 5px;"></div>'
            else:
                frame_data_html = "无帧数据"
            
            html_content += f"""
            <tr>
                <td>{fish_id}</td>
                <td>{tracking_duration:.1f}</td>
                <td>{track_distance_pixel:.1f}</td>
                <td>{track_distance_real:.1f}</td>
                <td>{avg_speed_real:.3f}</td>
                <td>{avg_fish_length_real:.1f}</td>
                <td>{avg_freq:.1f}</td>
                <td>{main_posture}</td>
                <td>{frame_data_html}</td>
            </tr>
            """
        
        html_content += """
        </table>
    </div>
    
    <script>
    // 帧数据的JavaScript数据
    const fishFrameData = {"""
        
        # 添加帧数据的JavaScript对象
        if hasattr(tracker, 'tracking_data'):
            for fish_id, tracking_entries in tracker.tracking_data.items():
                if tracking_entries:
                    html_content += f"""
        {fish_id}: ["""
                    for entry in tracking_entries:  # 显示所有帧
                        tail_pos_pixel = entry.get('tail_position_pixel', (0, 0))
                        tail_pos_real = entry.get('tail_position_real', (0, 0)) if entry.get('real_scale_available', False) else (0, 0)
                        
                        html_content += f"""
            {{
                'frame': {entry.get('frame', 0)},
                'timestamp': {entry.get('timestamp', 0):.3f},
                'tail_pos_pixel': [{tail_pos_pixel[0]:.1f}, {tail_pos_pixel[1]:.1f}],
                'tail_pos_real': [{tail_pos_real[0]:.1f}, {tail_pos_real[1]:.1f}],
                'tail_angle': {entry.get('tail_angle', 0):.1f},
                'tail_freq': {entry.get('tail_freq', 0)},
                'speed_pixel': {entry.get('speed_pixel', 0):.2f},
                'speed_real': {entry.get('speed_real', 0):.3f},
                'fish_length_pixel': {entry.get('fish_length_pixel', 0):.1f},
                'fish_length_real': {entry.get('fish_length_real', 0):.1f},
                'posture': '{entry.get('posture', 'Unknown')}',
                'corrected': {str(entry.get('corrected_available', False)).lower()},
                'real_scale': {str(entry.get('real_scale_available', False)).lower()}
            }},"""
                    html_content += """
        ],"""
        
        html_content += """
    };
    
    function showFrameData(fishId) {
        const selector = document.getElementById(`fish_${fishId}_frames`);
        const detailDiv = document.getElementById(`fish_${fishId}_detail`);
        const selectedFrameIndex = selector.value;
        
        if (selectedFrameIndex !== "" && fishFrameData[fishId] && fishFrameData[fishId][selectedFrameIndex]) {
            const data = fishFrameData[fishId][selectedFrameIndex];
                            detailDiv.innerHTML = `
                <strong>帧号 ${data.frame} 详细信息：</strong><br>
                <div style="display: grid; grid-template-columns: 1fr 1fr 1fr; gap: 10px; margin-top: 10px;">
                    <div>
                        <strong>基本信息：</strong><br>
                        时间戳: ${data.timestamp} 秒<br>
                        体态: ${data.posture}<br>
                        尾部角度: ${data.tail_angle}°<br>
                        尾摆频率: ${data.tail_freq} 次/5秒
                    </div>
                    <div>
                        <strong>位置(像素)：</strong><br>
                        X: ${data.tail_pos_pixel[0]}<br>
                        Y: ${data.tail_pos_pixel[1]}<br>
                        <strong>位置(毫米)：</strong><br>
                        X: ${data.tail_pos_real[0]}<br>
                        Y: ${data.tail_pos_real[1]}
                    </div>
                    <div>
                        <strong>运动参数：</strong><br>
                        速度(像素/秒): ${data.speed_pixel}<br>
                        速度(毫米/秒): ${data.speed_real}<br>
                        鱼体长度(像素): ${data.fish_length_pixel}<br>
                        鱼体长度(毫米): ${data.fish_length_real}
                    </div>
                    <div>
                        <strong>数据状态：</strong><br>
                        畸变矫正: ${data.corrected ? '是' : '否'}<br>
                        真实尺寸: ${data.real_scale ? '是' : '否'}
                    </div>
                </div>
            `;
            detailDiv.style.display = 'block';
        } else {
            detailDiv.style.display = 'none';
        }
    }
    </script>
</body>
</html>
        """
        
        return html_content
    
    def _generate_charts(self, fish_data, report_dir):
        """生成统计图表"""
        try:
            import matplotlib.pyplot as plt
            import matplotlib
            matplotlib.use('Agg')  # 非交互式后端
            
            # 解决中文显示问题 - 使用英文标签避免字体问题
            plt.rcParams['font.family'] = 'DejaVu Sans'
            
            # 速度分布图
            speeds = [info.get('average_speed_pixel', 0) for info in fish_data.values()]
            if speeds:
                plt.figure(figsize=(10, 6))
                plt.hist(speeds, bins=20, alpha=0.7, color='skyblue', edgecolor='black')
                plt.title('Fish Swimming Speed Distribution')
                plt.xlabel('Average Speed (pixels/frame)')
                plt.ylabel('Number of Fish')
                plt.grid(True, alpha=0.3)
                plt.savefig(os.path.join(report_dir, 'speed_distribution.png'), dpi=300, bbox_inches='tight')
                plt.close()
            
            # 尾摆频率分布图
            freqs = [info.get('tail_frequency', 0) / 5.0 for info in fish_data.values()]  # 转换为Hz
            if freqs:
                plt.figure(figsize=(10, 6))
                plt.hist(freqs, bins=20, alpha=0.7, color='lightgreen', edgecolor='black')
                plt.title('Fish Tail Beat Frequency Distribution')
                plt.xlabel('Tail Beat Frequency (Hz)')
                plt.ylabel('Number of Fish')
                plt.grid(True, alpha=0.3)
                plt.savefig(os.path.join(report_dir, 'frequency_distribution.png'), dpi=300, bbox_inches='tight')
                plt.close()
            
            # 轨迹长度分布图
            track_lengths = [len(info.get('positions', [])) for info in fish_data.values()]
            if track_lengths:
                plt.figure(figsize=(10, 6))
                plt.hist(track_lengths, bins=20, alpha=0.7, color='orange', edgecolor='black')
                plt.title('Fish Track Length Distribution')
                plt.xlabel('Track Length (frames)')
                plt.ylabel('Number of Fish')
                plt.grid(True, alpha=0.3)
                plt.savefig(os.path.join(report_dir, 'track_length_distribution.png'), dpi=300, bbox_inches='tight')
                plt.close()
                
        except ImportError:
            print("matplotlib未安装，跳过图表生成")
        except Exception as e:
            print(f"生成图表时出错: {e}")
    
    def _save_processed_video(self):
        """保存处理后的视频"""
        try:
            import cv2
            processed_frames = self.video_thread.processed_frames
            
            if not processed_frames:
                return
                
            # 创建输出目录
            output_dir = "realtime_output"
            os.makedirs(output_dir, exist_ok=True)
            
            # 设置视频编码器和输出文件
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            output_file = os.path.join(output_dir, f"tracked_video_{timestamp}.mp4")
            
            # 获取第一帧来确定视频尺寸
            first_frame = processed_frames[0]
            height, width = first_frame.shape[:2]
            
            # 创建视频写入器，使用原始视频帧率
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            fps = self.video_thread.original_fps  # 使用原视频帧率而不是固定30fps
            print(f"保存视频，使用原始帧率: {fps} fps")
            out = cv2.VideoWriter(output_file, fourcc, fps, (width, height))
            
            # 写入所有处理后的帧
            for frame in processed_frames:
                out.write(frame)
            
            out.release()
            
            self.log_text.append(f"处理后的视频已保存: {output_file}")
            self.log_text.append(f"总帧数: {len(processed_frames)}")
            
        except Exception as e:
            import traceback
            error_details = traceback.format_exc()
            print(f"保存视频详细错误: {error_details}")
            self.log_text.append(f"保存视频失败: {str(e)}")
             
    def closeEvent(self, event):
        """关闭事件"""
        if self.is_tracking:
            self.stop_tracking()
        event.accept()


def main():
    """主函数"""
    app = QApplication(sys.argv)
    
    # 设置应用信息
    app.setApplicationName("鱼类跟踪系统")
    app.setApplicationVersion("1.0")
    app.setOrganizationName("Fish Tracking")
    
    # 创建主窗口
    window = MainWindow()
    window.show()
    
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()