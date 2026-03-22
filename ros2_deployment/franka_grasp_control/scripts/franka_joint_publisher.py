#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState
import numpy as np
import time

class FrankaJointPublisher(Node):
    def __init__(self):
        super().__init__('franka_joint_publisher')
        self.pub = self.create_publisher(JointState, '/joint_command', 10)
        # 设置20Hz定时器，周期0.05秒
        self.timer = self.create_timer(0.05, self.timer_callback)
        self.start_time = time.time()
        
        self.get_logger().info("🦾 纯关节正弦波控制器已启动！准备展示太极关节舞...")

        # 初始化JointState消息对象
        self.joint_state = JointState()
        self.joint_state.name = [
            "panda_joint1",       # 关节1
            "panda_joint2",       # 关节2
            "panda_joint3",       # 关节3
            "panda_joint4",       # 关节4
            "panda_joint5",       # 关节5
            "panda_joint6",       # 关节6
            "panda_joint7",       # 关节7
            "panda_finger_joint1",# 手指关节1
            "panda_finger_joint2" # 手指关节2
        ]

        # 默认关节角度 (安全的起步微屈姿态)
        self.default_joints = np.array([0.0, -1.16, 0.0, -2.3, 0.0, 1.6, 1.1, 0.4, 0.4])
        
        # 🌟 修复 1：安全的振幅，绝对不会触发物理骨折和引擎崩溃
        self.amplitude = np.array([
            0.3,   # joint1：左右微调
            0.3,   # joint2：大臂微抬
            0.3,   # joint3：大臂扭转
            0.5,   # joint4：手肘微弯 (最大 -2.3 - 0.5 = -2.8，在极限内)
            0.3,   # joint5：小臂扭转
            0.5,   # joint6：手腕下压
            0.5,   # joint7：手腕旋转
            0.0,   # panda_finger_joint1：夹爪保持不动
            0.0    # panda_finger_joint2：夹爪保持不动
        ])

    def timer_callback(self):
        # 获取运行时间
        t = time.time() - self.start_time
        
        # 🌟 修复 2：减慢时间流速，0.5代表动作放慢一倍 (降频)
        omega = 0.5 
        
        # 使用正弦函数计算每个关节的新位置
        pos = np.sin(t * omega) * self.amplitude + self.default_joints
        
        # 更新消息头时间戳
        self.joint_state.header.stamp = self.get_clock().now().to_msg()
        self.joint_state.position = pos.tolist()
        
        self.pub.publish(self.joint_state)
        
        # 避免刷屏太快，只打印两个关键关节的角度看看趋势
        self.get_logger().info(f"太极舞中... 关节1: {pos[0]:.2f}, 关节4(手肘): {pos[3]:.2f}")

def main(args=None):
    rclpy.init(args=args)
    node = FrankaJointPublisher()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()