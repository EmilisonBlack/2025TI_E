from vision_detect import HSLDetector
import numpy as np

class LaserMotorController:
    def __init__(self, kp=1, ki=0.1, kd=0.4, max_output=100, min_output=-100):
        """
        初始化激光电机控制器
        
        参数:
            kp: 比例系数
            ki: 积分系数
            kd: 微分系数
            max_output: 最大输出限制
            min_output: 最小输出限制
        """
        
        #视觉检测实例初始化
        detector=HSLDetector()

        #矩形HSL阈值
        detector.h_min_rect, detector.h_max_rect = 0, 179
        detector.s_min_rect, detector.s_max_rect = 0, 255
        detector.l_min_rect, detector.l_max_rect = 0, 255
        
        #激光点HSL阈值
        self.h_min_laser, self.h_max_laser = 0, 179
        self.s_min_laser, self.s_max_laser = 0, 255
        self.l_min_laser, self.l_max_laser = 0, 255
        
        # 检测参数
        detector.area_threshold = 600  # 连通域面积阈值
        detector.angle_min =60        # 最小角度阈值(度)
        detector.angle_max = 120       # 最大角度阈值(度)



        # PID参数
        self.kp = kp
        self.ki = ki
        self.kd = kd
        
        # 输出限制，防止电机控制信号过大
        self.max_output = max_output
        self.min_output = min_output
        
        # PID变量初始化
        self.prev_error_x = 0
        self.prev_error_y = 0
        self.integral_x = 0
        self.integral_y = 0
        
        # 目标中心点
        self.target_center = (0, 0)
        
    def extract_coordinates(self, results):
        """
        从结果中提取角点和激光点坐标
        
        参数:
            results: 包含角点和激光点的字典列表
            
        返回:
            corners: 4个角点的坐标列表
            laser_point: 激光点坐标，若不存在则为None
        """
        corners = []
        laser_point = None
        
        for item in results:
            if item['type'] == 'corner':
                corners.append(item['coordinates'])
            elif item['type'] == 'laser':
                laser_point = item['coordinates']
        
        # 确保我们有4个角点
        if len(corners) != 4:
            raise ValueError(f"期望4个角点，但收到{len(corners)}个")
            
        return corners, laser_point
    
    def calculate_center(self, corners):
        """
        根据4个角点计算矩形中心点
        
        参数:
            corners: 4个角点的坐标列表
            
        返回:
            center: 矩形中心点坐标(x, y)
        """
        # 提取x和y坐标
        x_coords = [point[0] for point in corners]
        y_coords = [point[1] for point in corners]
        
        # 计算中心点（矩形对角线交点）
        center_x = (min(x_coords) + max(x_coords)) / 2
        center_y = (min(y_coords) + max(y_coords)) / 2
        
        self.target_center = (center_x, center_y)
        return self.target_center
    
    def pid_control(self, current_pos, target_pos, dt=0.1):
        """
        PID控制算法
        
        参数:
            current_pos: 当前位置(x, y)
            target_pos: 目标位置(x, y)
            dt: 时间间隔
            
        返回:
            (output_x, output_y): x和y方向的控制输出
        """
        # 计算位置误差
        error_x = target_pos[0] - current_pos[0]
        error_y = target_pos[1] - current_pos[1]
        
        # 计算积分项
        self.integral_x += error_x * dt
        self.integral_y += error_y * dt
        
        # 防止积分饱和
        self.integral_x = self._clamp(self.integral_x)
        self.integral_y = self._clamp(self.integral_y)
        
        # 计算微分项
        derivative_x = (error_x - self.prev_error_x) / dt if dt > 0 else 0
        derivative_y = (error_y - self.prev_error_y) / dt if dt > 0 else 0
        
        # 计算PID输出
        output_x = self.kp * error_x + self.ki * self.integral_x + self.kd * derivative_x
        output_y = self.kp * error_y + self.ki * self.integral_y + self.kd * derivative_y
        
        # 保存当前误差用于下次计算微分
        self.prev_error_x = error_x
        self.prev_error_y = error_y
        
        # 限制输出范围
        output_x = self._clamp(output_x)
        output_y = self._clamp(output_y)
        
        return output_x, output_y
    
    def _clamp(self, value):
        """限制值在最大和最小输出之间"""
        return max(min(value, self.max_output), self.min_output)
    
    def generate_motor_commands(self, output_x, output_y):
        """
        根据PID输出生成电机控制指令
        
        参数:
            output_x: x方向的PID输出
            output_y: y方向的PID输出
            
        返回:
            电机控制指令字典
        """
        # 这里假设我们有两个电机，分别控制x和y方向
        # 实际应用中，可能需要根据电机布局进行坐标转换
        return {
            'motor_x': output_x,
            'motor_y': output_y,
            'direction_x': 'right' if output_x > 0 else 'left',
            'direction_y': 'down' if output_y > 0 else 'up',  # 计算机坐标系中y向下为正
            'magnitude_x': abs(output_x),
            'magnitude_y': abs(output_y)
        }
    
    def update(self, results, dt=0.1):
        """
        更新控制系统，处理新的传感器数据并生成控制指令
        
        参数:
            results: 包含角点和激光点的字典列表
            dt: 与上一次更新的时间间隔
            
        返回:
            电机控制指令，若激光点不存在则返回None
        """
        try:
            # 提取坐标
            corners, laser_point = self.extract_coordinates(results)
            
            # 激光点不存在时返回None
            if laser_point is None:
                print("未检测到激光点")
                return None
                
            # 计算目标中心点
            target_center = self.calculate_center(corners)
            
            # 计算控制输出
            output_x, output_y = self.pid_control(laser_point, target_center, dt)
            
            # 生成电机指令
            commands = self.generate_motor_commands(output_x, output_y)
            
            # 打印状态信息（实际应用中可删除或改为日志）
            print(f"目标中心点: {target_center}")
            print(f"激光点位置: {laser_point}")
            print(f"控制指令: {commands}")
            
            return commands
            
        except Exception as e:
            print(f"控制更新失败: {str(e)}")
            return None


# 示例用法
if __name__ == "__main__":
    # 创建控制器实例，可根据实际情况调整PID参数
    controller = LaserMotorController(kp=0.8, ki=0.05, kd=0.3)
    
    # 示例数据 - 第一帧
    sample_results_1 = [
        {'type': 'corner', 'coordinates': (50, 50)},
        {'type': 'corner', 'coordinates': (150, 50)},
        {'type': 'corner', 'coordinates': (50, 150)},
        {'type': 'corner', 'coordinates': (150, 150)},
        {'type': 'laser', 'coordinates': (20, 30)}  # 激光点在左上角
    ]
    
    # 处理第一帧数据
    print("处理第一帧数据:")
    commands_1 = controller.update(sample_results_1)
    
    # 示例数据 - 第二帧（激光点移动了一些）
    sample_results_2 = [
        {'type': 'corner', 'coordinates': (50, 50)},
        {'type': 'corner', 'coordinates': (150, 50)},
        {'type': 'corner', 'coordinates': (50, 150)},
        {'type': 'corner', 'coordinates': (150, 150)},
        {'type': 'laser', 'coordinates': (60, 70)}  # 激光点向中心移动了一些
    ]
    
    # 处理第二帧数据
    print("\n处理第二帧数据:")
    commands_2 = controller.update(sample_results_2)
    
    # 示例数据 - 第三帧（激光点接近中心）
    sample_results_3 = [
        {'type': 'corner', 'coordinates': (50, 50)},
        {'type': 'corner', 'coordinates': (150, 50)},
        {'type': 'corner', 'coordinates': (50, 150)},
        {'type': 'corner', 'coordinates': (150, 150)},
        {'type': 'laser', 'coordinates': (95, 98)}  # 激光点接近中心
    ]
    
    # 处理第三帧数据
    print("\n处理第三帧数据:")
    commands_3 = controller.update(sample_results_3)
