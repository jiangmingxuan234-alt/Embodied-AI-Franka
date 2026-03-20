import argparse
from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="ROS2 Joint Environment Bridge")
AppLauncher.add_app_launcher_args(parser)
parser.set_defaults(enable_cameras=False)
args_cli = parser.parse_args()
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import omni.kit.app
ext_manager = omni.kit.app.get_app().get_extension_manager()
ext_manager.set_extension_enabled_immediate("omni.isaac.ros2_bridge", True)

import torch
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState # 🌟 换回 JointState
from isaaclab.envs import ManagerBasedRLEnv
from peg_in_hole.tasks.manipulation.peg_in_hole.env_cfg import PegInHoleEnvCfg

global_joint_targets = None

class JointSubscriber(Node):
    def __init__(self):
        super().__init__('isaac_lab_joint_bridge')
        # 🌟 监听 /joint_command 话题
        self.subscription = self.create_subscription(
            JointState, '/joint_command', self.listener_callback, 10)

    def listener_callback(self, msg):
        global global_joint_targets
        # 直接把收到的 7 个角度存下来
        global_joint_targets = torch.tensor(
            [msg.position[:7]], dtype=torch.float32, device='cuda:0')

def main():
    global global_joint_targets
    
    # 1. 加载配置
    env_cfg = PegInHoleEnvCfg()
    env_cfg.scene.num_envs = 1 
    
    # 🌟 终极破解：打破 RL 训练引擎的回合制重置诅咒！
    # 将每回合的最大时间强行拉长到 10000 秒（约 2.7 小时），让它永不自动刷新场景！
    env_cfg.episode_length_s = 10000.0 

    # 2. 实例化环境
    env = ManagerBasedRLEnv(cfg=env_cfg)
    action = torch.zeros((1, 8), device=env.device)
    
    # 3. 初始化 ROS 2 节点
    rclpy.init()
    ros_node = JointSubscriber()
    
    env.reset()
    print("\n✅ 纯关节驱动桥接启动！(已解除强制重置时间限制)")
    print("当前状态：[安全锁死等待]")
    print("正在监听 /joint_command (JointState 角度) ...\n")
    
    # 安全待命姿态 (高举避让，不碰桌面)
    safe_joint_pos = torch.tensor([[0.0, -1.0, 0.0, -2.0, 0.0, 1.57, 0.78]], dtype=torch.float32, device=env.device)

    # 4. 主循环
    while simulation_app.is_running():
        # 处理 ROS 2 消息回调
        rclpy.spin_once(ros_node, timeout_sec=0.0)
        
        # 🌟 收到指令就直接赋给电机，没收到就死死锁在安全姿态
        if global_joint_targets is not None:
            action[:, 0:7] = global_joint_targets
        else:
            action[:, 0:7] = safe_joint_pos
            
        action[:, 7] = 1.0 # 夹爪保持张开
        
        # 步进物理环境
        env.step(action)

    # 5. 清理退出
    ros_node.destroy_node()
    rclpy.shutdown()

if __name__ == "__main__":
    main()
    simulation_app.close()