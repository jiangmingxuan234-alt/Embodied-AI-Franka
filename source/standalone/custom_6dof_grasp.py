import argparse
from isaaclab.app import AppLauncher
import numpy as np

# 启动配置
parser = argparse.ArgumentParser(description="Pure IK 6-DoF Grasping (No RMPFlow)")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

# ==========================================
import omni.kit.app
ext_manager = omni.kit.app.get_app().get_extension_manager()
ext_manager.set_extension_enabled_immediate("omni.isaac.core", True)
ext_manager.set_extension_enabled_immediate("omni.isaac.franka", True)

from isaacsim.core.api import World
from isaacsim.core.api.objects import DynamicCuboid
from isaacsim.core.api.materials import PhysicsMaterial

# 🌟 修复警告 1：使用 4.5.0 最新的 Franka 路径
from isaacsim.robot.manipulators.examples.franka import Franka
from omni.isaac.franka.kinematics_solver import KinematicsSolver
# 🌟 修复警告 2：使用最新的 rotations 路径
from isaacsim.core.utils.rotations import euler_angles_to_quat

def main():
    world = World(stage_units_in_meters=1.0, physics_dt=1.0/120.0, rendering_dt=1.0/30.0)
    world.scene.add_default_ground_plane()

    franka = world.scene.add(Franka(prim_path="/World/Fancy_Franka", name="fancy_franka"))
    high_friction = PhysicsMaterial("/World/Physics_Materials/HighFriction", static_friction=2.0, dynamic_friction=2.0, restitution=0.0)

    # 生成旋转了 45 度的 U 盘
    yaw_angle_rad = np.pi / 4 
    u_disk_quat = euler_angles_to_quat(np.array([0.0, 0.0, yaw_angle_rad]))

    u_disk = world.scene.add(
        DynamicCuboid(
            prim_path="/World/u_disk",
            name="u_disk",
            position=np.array([0.45, -0.2, 0.02]), 
            orientation=u_disk_quat, 
            scale=np.array([0.12, 0.03, 0.025]), 
            color=np.array([1.0, 0.0, 0.0]),         
            mass=0.02,
            physics_material=high_friction 
        )
    )

    world.reset()
    init_joints = np.array([0.0, -0.785, 0.0, -2.356, 0.0, 1.57, 0.785, 0.04, 0.04])
    franka.set_joint_positions(init_joints)

    ik_solver = KinematicsSolver(franka)
    phase = "approach"

    target_quat = euler_angles_to_quat(np.array([np.pi, 0.0, yaw_angle_rad]))

    print("\n🚀 纯 IK 抓取启动！内存只读 Bug 已修复。")

    while simulation_app.is_running():
        world.step(render=True)
        
        if world.is_playing():
            if world.current_time_step_index == 0:
                world.reset()
                franka.set_joint_positions(init_joints)
                phase = "approach"

            curr_u_pos, _ = u_disk.get_world_pose()
            
            # =============== 纯状态机 ===============
            if phase == "approach":
                target_pos = curr_u_pos + np.array([0.0, 0.0, 0.10])
                
                actions, success = ik_solver.compute_inverse_kinematics(
                    target_position=target_pos,
                    target_orientation=target_quat
                )
                
                if success:
                    # 🌟 核心修复：复制一份数组，使其变成可写的！
                    mutable_joints = np.array(actions.joint_positions)
                    mutable_joints[-2:] = [0.04, 0.04] # 张开夹爪
                    actions.joint_positions = mutable_joints # 塞回去
                    franka.apply_action(actions)
                
                dist = np.linalg.norm(franka.end_effector.get_world_pose()[0] - target_pos)
                if dist < 0.02:
                    print("🔄 到达预抓取点，准备下潜...")
                    phase = "grasp"
                    
            elif phase == "grasp":
                target_pos = curr_u_pos + np.array([0.0, 0.0, 0.015])
                
                actions, success = ik_solver.compute_inverse_kinematics(
                    target_position=target_pos,
                    target_orientation=target_quat
                )
                
                if success:
                    # 🌟 同样使用可写数组
                    mutable_joints = np.array(actions.joint_positions)
                    mutable_joints[-2:] = [0.04, 0.04]
                    actions.joint_positions = mutable_joints
                    franka.apply_action(actions)
                
                dist = np.linalg.norm(franka.end_effector.get_world_pose()[0] - target_pos)
                if dist < 0.02:
                    print("✊ 接触目标！执行强制闭合...")
                    phase = "lift"
                    
            elif phase == "lift":
                target_pos = np.array([0.45, 0.0, 0.40]) 
                
                actions, success = ik_solver.compute_inverse_kinematics(
                    target_position=target_pos,
                    target_orientation=target_quat
                )
                
                if success:
                    # 🌟 闭合锁死
                    mutable_joints = np.array(actions.joint_positions)
                    mutable_joints[-2:] = [0.0, 0.0] 
                    actions.joint_positions = mutable_joints
                    franka.apply_action(actions)

if __name__ == '__main__':
    main()
    simulation_app.close()