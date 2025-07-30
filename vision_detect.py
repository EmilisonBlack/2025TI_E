import cv2
import numpy as np
import tkinter as tk
from tkinter import ttk
from PIL import Image, ImageTk
import threading
import time
import math

class HSLDetector:
    def __init__(self):
        # 初始化HSL阈值
        self.h_min_rect, self.h_max_rect = 0, 179
        self.s_min_rect, self.s_max_rect = 0, 255
        self.l_min_rect, self.l_max_rect = 0, 255
        
        self.h_min_laser, self.h_max_laser = 0, 179
        self.s_min_laser, self.s_max_laser = 0, 255
        self.l_min_laser, self.l_max_laser = 0, 255
        
        # 检测参数
        self.area_threshold = 600  # 连通域面积阈值
        self.angle_min =60        # 最小角度阈值(度)
        self.angle_max = 120       # 最大角度阈值(度)
        
        # 检测结果
        self.corner_points = []
        self.laser_point = None
        
        # 视频相关
        self.cap = None
        self.running = False
        self.fps = 30
        self.frame_interval = 1.0 / self.fps
        
        # GUI相关
        self.root = None
        self.gui_running = False

    def set_fps(self, fps):
        """设置帧率"""
        self.fps = max(1, min(60, fps))
        self.frame_interval = 1.0 / self.fps

    def process_frame(self, frame):
        """处理单帧图像，返回处理结果"""
        # 转换为HSL色彩空间
        hsl = cv2.cvtColor(frame, cv2.COLOR_BGR2HLS)
        
        # 直角点检测的二值化图像
        lower_rect = np.array([self.h_min_rect, self.l_min_rect, self.s_min_rect])
        upper_rect = np.array([self.h_max_rect, self.l_max_rect, self.s_max_rect])
        mask_rect = cv2.inRange(hsl, lower_rect, upper_rect)
        
        # 激光点检测的二值化图像
        lower_laser = np.array([self.h_min_laser, self.l_min_laser, self.s_min_laser])
        upper_laser = np.array([self.h_max_laser, self.l_max_laser, self.s_max_laser])
        mask_laser = cv2.inRange(hsl, lower_laser, upper_laser)
        
        # 检测直角点
        self.detect_corner_points(mask_rect)
        
        # 检测激光点
        self.detect_laser_point(mask_laser)
        
        # 在原图上绘制检测结果
        result_frame = frame.copy()
        for (x, y) in self.corner_points:
            cv2.circle(result_frame, (x, y), 5, (0, 255, 0), -1)
            cv2.putText(result_frame, f"Corner {self.corner_points.index((x, y))+1}", 
                        (x+10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        
        if self.laser_point:
            cv2.circle(result_frame, self.laser_point, 5, (0, 0, 255), -1)
            cv2.putText(result_frame, "Laser", 
                        (self.laser_point[0]+10, self.laser_point[1]), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
        
        return mask_rect, mask_laser, result_frame

    def angle_between_points(self, p1, p2, p3):
        """计算三个点形成的角度（p2为顶点）"""
        # 向量计算
        v1 = np.array(p1) - np.array(p2)
        v2 = np.array(p3) - np.array(p2)
        
        # 计算夹角余弦值
        dot_product = np.dot(v1, v2)
        norm_v1 = np.linalg.norm(v1)
        norm_v2 = np.linalg.norm(v2)
        
        if norm_v1 == 0 or norm_v2 == 0:
            return 0.0
            
        cos_angle = dot_product / (norm_v1 * norm_v2)
        # 防止数值计算导致的微小溢出
        cos_angle = max(min(cos_angle, 1.0), -1.0)
        
        # 转换为角度
        angle_rad = np.arccos(cos_angle)
        angle_deg = np.degrees(angle_rad)
        
        return angle_deg

    def detect_corner_points(self, mask):
        """检测直角点，只检测四边形并考虑70-110度范围的角度"""
        self.corner_points = []
        
        # 寻找连通域
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        for contour in contours:
            # 应用面积阈值过滤小连通域
            area = cv2.contourArea(contour)
            if area < self.area_threshold:
                continue
            
            # 多边形逼近
            perimeter = cv2.arcLength(contour, True)
            # 根据周长调整逼近精度
            epsilon = 0.02 * perimeter
            approx = cv2.approxPolyDP(contour, epsilon, True)
            
            # 只检测四边形
            if len(approx) == 4:
                # 提取顶点坐标
                points = [tuple(point[0]) for point in approx]
                valid_corners = []
                
                # 检查每个顶点的角度
                for i in range(len(points)):
                    # 前一个点和后一个点（循环）
                    p_prev = points[i-1]
                    p_curr = points[i]
                    p_next = points[(i+1) % len(points)]
                    
                    # 计算角度
                    angle = self.angle_between_points(p_prev, p_curr, p_next)
                    
                    # 检查角度是否在有效范围内
                    if self.angle_min <= angle <= self.angle_max:
                        valid_corners.append(p_curr)
                
                # 如果找到有效角点，添加到结果
                if valid_corners:
                    self.corner_points.extend(valid_corners)
                    # 限制最多4个角点
                    if len(self.corner_points) >= 4:
                        self.corner_points = self.corner_points[:4]
                        break
        
        # 确保角点数量不超过4个
        self.corner_points = self.corner_points[:4]

    def detect_laser_point(self, mask):
        """检测激光点，不进行面积过滤"""
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        self.laser_point = None
        
        max_area = 0
        best_contour = None
        
        # 不对激光点进行面积过滤，只选择最大面积的轮廓
        for contour in contours:
            area = cv2.contourArea(contour)
            if area > max_area:
                max_area = area
                best_contour = contour
        
        if best_contour is not None:
            moments = cv2.moments(best_contour)
            if moments["m00"] != 0:
                cX = int(moments["m10"] / moments["m00"])
                cY = int(moments["m01"] / moments["m00"])
                self.laser_point = (cX, cY)

    def get_detection_results(self):
        """返回检测结果数组"""
        result = []
        # 添加角点坐标
        for point in self.corner_points:
            result.append({"type": "corner", "coordinates": point})
        # 添加激光点坐标
        if self.laser_point:
            result.append({"type": "laser", "coordinates": self.laser_point})
        return result

    def start_camera(self):
        """启动摄像头"""
        # 使用1号摄像头
        self.cap = cv2.VideoCapture(1)
        if not self.cap.isOpened():
            raise Exception("无法打开摄像头，请检查设备是否连接正常")

    def stop_camera(self):
        """停止摄像头"""
        if self.cap is not None:
            self.cap.release()
        self.running = False

    def run_detection(self, callback=None):
        """运行检测（无GUI模式）"""
        self.start_camera()
        self.running = True
        
        try:
            while self.running:
                start_time = time.time()
                ret, frame = self.cap.read()
                if not ret:
                    break
                
                mask_rect, mask_laser, result_frame = self.process_frame(frame)
                
                # 调用回调函数处理结果
                if callback:
                    callback(mask_rect, mask_laser, result_frame, self.get_detection_results())
                
                # 控制帧率
                elapsed_time = time.time() - start_time
                sleep_time = self.frame_interval - elapsed_time
                if sleep_time > 0:
                    time.sleep(sleep_time)
        finally:
            self.stop_camera()

    def _create_sliders(self, parent, label_text, h_min, h_max, s_min, s_max, l_min, l_max, update_callback):
        """创建HSL滑块控件"""
        frame = ttk.LabelFrame(parent, text=label_text)
        frame.pack(fill="x", padx=5, pady=5)
        
        # H通道
        ttk.Label(frame, text="色调(H)范围:").grid(row=0, column=0, sticky="w", padx=5)
        h_min_slider = ttk.Scale(frame, from_=0, to=179, value=h_min, command=lambda v: update_callback('h_min', float(v)))
        h_min_slider.grid(row=0, column=1, sticky="ew", padx=5)
        h_min_label = ttk.Label(frame, text=str(h_min))
        h_min_label.grid(row=0, column=2, padx=5)
        
        h_max_slider = ttk.Scale(frame, from_=0, to=179, value=h_max, command=lambda v: update_callback('h_max', float(v)))
        h_max_slider.grid(row=0, column=3, sticky="ew", padx=5)
        h_max_label = ttk.Label(frame, text=str(h_max))
        h_max_label.grid(row=0, column=4, padx=5)
        
        # S通道
        ttk.Label(frame, text="饱和度(S)范围:").grid(row=1, column=0, sticky="w", padx=5)
        s_min_slider = ttk.Scale(frame, from_=0, to=255, value=s_min, command=lambda v: update_callback('s_min', float(v)))
        s_min_slider.grid(row=1, column=1, sticky="ew", padx=5)
        s_min_label = ttk.Label(frame, text=str(s_min))
        s_min_label.grid(row=1, column=2, padx=5)
        
        s_max_slider = ttk.Scale(frame, from_=0, to=255, value=s_max, command=lambda v: update_callback('s_max', float(v)))
        s_max_slider.grid(row=1, column=3, sticky="ew", padx=5)
        s_max_label = ttk.Label(frame, text=str(s_max))
        s_max_label.grid(row=1, column=4, padx=5)
        
        # L通道
        ttk.Label(frame, text="亮度(L)范围:").grid(row=2, column=0, sticky="w", padx=5)
        l_min_slider = ttk.Scale(frame, from_=0, to=255, value=l_min, command=lambda v: update_callback('l_min', float(v)))
        l_min_slider.grid(row=2, column=1, sticky="ew", padx=5)
        l_min_label = ttk.Label(frame, text=str(l_min))
        l_min_label.grid(row=2, column=2, padx=5)
        
        l_max_slider = ttk.Scale(frame, from_=0, to=255, value=l_max, command=lambda v: update_callback('l_max', float(v)))
        l_max_slider.grid(row=2, column=3, sticky="ew", padx=5)
        l_max_label = ttk.Label(frame, text=str(l_max))
        l_max_label.grid(row=2, column=4, padx=5)
        
        # 配置列权重，使滑块可以拉伸
        frame.columnconfigure(1, weight=1)
        frame.columnconfigure(3, weight=1)
        
        return {
            'h_min_slider': h_min_slider, 'h_min_label': h_min_label,
            'h_max_slider': h_max_slider, 'h_max_label': h_max_label,
            's_min_slider': s_min_slider, 's_min_label': s_min_label,
            's_max_slider': s_max_slider, 's_max_label': s_max_label,
            'l_min_slider': l_min_slider, 'l_min_label': l_min_label,
            'l_max_slider': l_max_slider, 'l_max_label': l_max_label
        }

    def _update_rect_sliders(self, param, value):
        """更新直角点检测的HSL滑块值"""
        value = int(value)
        if param == 'h_min':
            self.h_min_rect = value
            self.rect_sliders['h_min_label']['text'] = str(value)
        elif param == 'h_max':
            self.h_max_rect = value
            self.rect_sliders['h_max_label']['text'] = str(value)
        elif param == 's_min':
            self.s_min_rect = value
            self.rect_sliders['s_min_label']['text'] = str(value)
        elif param == 's_max':
            self.s_max_rect = value
            self.rect_sliders['s_max_label']['text'] = str(value)
        elif param == 'l_min':
            self.l_min_rect = value
            self.rect_sliders['l_min_label']['text'] = str(value)
        elif param == 'l_max':
            self.l_max_rect = value
            self.rect_sliders['l_max_label']['text'] = str(value)

    def _update_laser_sliders(self, param, value):
        """更新激光点检测的HSL滑块值"""
        value = int(value)
        if param == 'h_min':
            self.h_min_laser = value
            self.laser_sliders['h_min_label']['text'] = str(value)
        elif param == 'h_max':
            self.h_max_laser = value
            self.laser_sliders['h_max_label']['text'] = str(value)
        elif param == 's_min':
            self.s_min_laser = value
            self.laser_sliders['s_min_label']['text'] = str(value)
        elif param == 's_max':
            self.s_max_laser = value
            self.laser_sliders['s_max_label']['text'] = str(value)
        elif param == 'l_min':
            self.l_min_laser = value
            self.laser_sliders['l_min_label']['text'] = str(value)
        elif param == 'l_max':
            self.l_max_laser = value
            self.laser_sliders['l_max_label']['text'] = str(value)

    def _update_fps(self, value):
        """更新帧率 - 修复转换错误"""
        # 先将字符串转换为浮点数，再转换为整数
        self.set_fps(int(float(value)))
        self.fps_label['text'] = f"帧率: {self.fps} FPS"

    def _update_area_threshold(self, value):
        """更新连通域面积阈值"""
        self.area_threshold = int(float(value))
        self.area_label['text'] = f"连通域面积阈值: {self.area_threshold}"

    def _update_gui_images(self, mask_rect, mask_laser, result_frame):
        """更新GUI中的图像显示"""
        # 转换为RGB格式以便Tkinter显示
        mask_rect_rgb = cv2.cvtColor(mask_rect, cv2.COLOR_GRAY2RGB)
        mask_laser_rgb = cv2.cvtColor(mask_laser, cv2.COLOR_GRAY2RGB)
        result_frame_rgb = cv2.cvtColor(result_frame, cv2.COLOR_BGR2RGB)
        
        # 调整图像大小以适应显示
        display_size = (400, 300)
        mask_rect_img = cv2.resize(mask_rect_rgb, display_size)
        mask_laser_img = cv2.resize(mask_laser_rgb, display_size)
        result_img = cv2.resize(result_frame_rgb, display_size)
        
        # 转换为PhotoImage
        self.rect_img_tk = ImageTk.PhotoImage(image=Image.fromarray(mask_rect_img))
        self.laser_img_tk = ImageTk.PhotoImage(image=Image.fromarray(mask_laser_img))
        self.result_img_tk = ImageTk.PhotoImage(image=Image.fromarray(result_img))
        
        # 更新标签图像
        self.rect_label.config(image=self.rect_img_tk)
        self.laser_label.config(image=self.laser_img_tk)
        self.result_label.config(image=self.result_img_tk)
        
        # 更新结果文本
        results = self.get_detection_results()
        result_text = "检测结果:\n"
        for item in results:
            type_text = "角点" if item['type'] == "corner" else "激光点"
            result_text += f"{type_text}: {item['coordinates']}\n"
        self.result_text.config(text=result_text)

    def _gui_loop(self):
        """GUI循环"""
        def update_callback(mask_rect, mask_laser, result_frame, results):
            # 在主线程中更新GUI
            self.root.after(0, self._update_gui_images, mask_rect, mask_laser, result_frame)
        
        # 启动检测线程
        self.detection_thread = threading.Thread(target=self.run_detection, args=(update_callback,), daemon=True)
        self.detection_thread.start()
        
        # 启动GUI主循环
        self.root.mainloop()
        
        # GUI关闭后停止检测
        self.stop_camera()
        self.gui_running = False

    def show_gui(self):
        """显示GUI界面"""
        if self.gui_running:
            return
            
        self.gui_running = True
        self.root = tk.Tk()
        self.root.title("HSL检测系统")
        # 确保中文显示正常
        self.root.option_add("*Font", "SimHei 10")
        
        # 创建图像显示区域
        image_frame = ttk.Frame(self.root)
        image_frame.pack(fill="both", expand=True, padx=10, pady=10)
        
        # 直角点检测的二值化图像
        rect_frame = ttk.LabelFrame(image_frame, text="直角点检测二值化图像")
        rect_frame.grid(row=0, column=0, padx=5, pady=5, sticky="nsew")
        self.rect_label = ttk.Label(rect_frame)
        self.rect_label.pack(fill="both", expand=True)
        
        # 激光点检测的二值化图像
        laser_frame = ttk.LabelFrame(image_frame, text="激光点检测二值化图像")
        laser_frame.grid(row=0, column=1, padx=5, pady=5, sticky="nsew")
        self.laser_label = ttk.Label(laser_frame)
        self.laser_label.pack(fill="both", expand=True)
        
        # 结果图像
        result_frame = ttk.LabelFrame(image_frame, text="检测结果叠加图像")
        result_frame.grid(row=0, column=2, padx=5, pady=5, sticky="nsew")
        self.result_label = ttk.Label(result_frame)
        self.result_label.pack(fill="both", expand=True)
        
        # 配置网格权重，使图像区域可以拉伸
        image_frame.columnconfigure(0, weight=1)
        image_frame.columnconfigure(1, weight=1)
        image_frame.columnconfigure(2, weight=1)
        image_frame.rowconfigure(0, weight=1)
        
        # 创建滑块控制区域
        control_frame = ttk.Frame(self.root)
        control_frame.pack(fill="x", padx=10, pady=5)
        
        # 直角点检测的HSL滑块
        self.rect_sliders = self._create_sliders(
            control_frame, "直角点检测HSL阈值调整",
            self.h_min_rect, self.h_max_rect,
            self.s_min_rect, self.s_max_rect,
            self.l_min_rect, self.l_max_rect,
            self._update_rect_sliders
        )
        
        # 激光点检测的HSL滑块
        self.laser_sliders = self._create_sliders(
            control_frame, "激光点检测HSL阈值调整",
            self.h_min_laser, self.h_max_laser,
            self.s_min_laser, self.s_max_laser,
            self.l_min_laser, self.l_max_laser,
            self._update_laser_sliders
        )
        
        # 连通域面积阈值控制
        area_control_frame = ttk.LabelFrame(self.root, text="连通域过滤")
        area_control_frame.pack(fill="x", padx=10, pady=5)
        
        ttk.Label(area_control_frame, text="面积阈值调整:").pack(side="left", padx=5)
        self.area_slider = ttk.Scale(area_control_frame, from_=10, to=1000, value=self.area_threshold, 
                                    command=lambda v: self._update_area_threshold(v))
        self.area_slider.pack(side="left", fill="x", expand=True, padx=5)
        self.area_label = ttk.Label(area_control_frame, text=f"连通域面积阈值: {self.area_threshold}")
        self.area_label.pack(side="left", padx=5)
        
        # 帧率控制
        fps_control_frame = ttk.LabelFrame(self.root, text="帧率控制")
        fps_control_frame.pack(fill="x", padx=10, pady=5)
        
        ttk.Label(fps_control_frame, text="帧率调整:").pack(side="left", padx=5)
        self.fps_slider = ttk.Scale(fps_control_frame, from_=1, to=60, value=self.fps, command=lambda v: self._update_fps(v))
        self.fps_slider.pack(side="left", fill="x", expand=True, padx=5)
        self.fps_label = ttk.Label(fps_control_frame, text=f"帧率: {self.fps} FPS")
        self.fps_label.pack(side="left", padx=5)
        
        # 结果显示区域
        result_text_frame = ttk.LabelFrame(self.root, text="检测结果")
        result_text_frame.pack(fill="x", padx=10, pady=5)
        self.result_text = ttk.Label(result_text_frame, text="等待检测...", justify="left")
        self.result_text.pack(fill="x", padx=5, pady=5)
        
        # 关闭按钮
        btn_frame = ttk.Frame(self.root)
        btn_frame.pack(fill="x", padx=10, pady=10)
        ttk.Button(btn_frame, text="退出", command=self.root.quit).pack(side="right")
        
        # 启动GUI循环
        self._gui_loop()

if __name__ == "__main__":
    # 单独运行时显示GUI界面
    detector = HSLDetector()
    detector.show_gui()
    