"""
IK-based full grasp data collection — 完整下压轨迹
用 KinematicsSolver 替代 PickPlaceController，确保 ee 真正下压到物体表面。

阶段：
  1. approach: ee 移动到物体正上方 10cm
  2. descend:  ee 从 10cm 垂直下压到物体表面 1.5cm
  3. close:    闭合夹爪
  4. lift:     抬起到目标高度

用法: python source/standalone/collect_ik_grasp.py --n_demos 300
"""
import argparse
from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
AppLauncher.add_app_launcher_args(parser)
parser.add_argument("--n_demos", type=int, default=300)
parser.add_argument("--output", type=str,
    default="/home/jmx001/my_program/my_robot_project/logs/robomimic/full_grasp_ik_v2.hdf5")
args_cli = parser.parse_args()
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import omni.kit.app
ext_manager = omni.kit.app.get_app().get_extension_manager()
ext_manager.set_extension_enabled_immediate("omni.isaac.core", True)
ext_manager.set_extension_enabled_immediate("omni.isaac.franka", True)

import numpy as np
import h5py
import json
import os

from omni.isaac.core import World
from omni.isaac.core.objects import DynamicCuboid
from omni.isaac.core.utils.types import ArticulationAction
from omni.isaac.franka import Franka
from omni.isaac.franka.kinematics_solver import KinematicsSolver

OBJ_X_RANGE = (0.42, 0.58)
OBJ_Y_RANGE = (-0.30, 0.30)
GOAL = np.array([0.40, 0.0, 0.40], dtype=np.float32)
INIT_JOINTS = np.array([0.0, -1.1, 0.0, -2.3, 0.0, 2.4, 0.8, 0.04, 0.04], dtype=np.float32)
MAX_STEPS = 1500
MIN_FRAMES = 80
RECORD_STRIDE = 2

# 下压目标：ee 到物体上方 1.5cm（和 custom_6dof_grasp.py 一致）
GRASP_Z_OFFSET = 0.015
# 预抓取高度：物体上方 10cm
APPROACH_Z_OFFSET = 0.10
# 到位阈值
APPROACH_THRESH = 0.02    # approach→descend
DESCEND_THRESH = 0.01     # descend→close
CARTESIAN_STEP = 0.002    # 每步最大笛卡尔位移 2mm
JOINT_SLEW = 0.05         # 每步最大关节变化 rad
# 夹爪闭合等待步数
CLOSE_WAIT = 40
# 抬起到位阈值
LIFT_THRESH = 0.03

# 末端执行器朝下的姿态 (euler [pi, 0, 0] → quat [w,x,y,z])
# rotation of pi around x-axis: w=cos(pi/2)=0, x=sin(pi/2)=1, y=0, z=0
EE_QUAT = np.array([0.0, 1.0, 0.0, 0.0], dtype=np.float32)


def build_obs(j_pos, j_vel, ee_pos, ee_quat, obj_pos):
    rel = obj_pos - ee_pos
    goal_rel = GOAL - obj_pos
    if obj_pos[2] <= 0.09:
        phase = np.array([1.0, 0.0, 0.0], dtype=np.float32)
    elif np.linalg.norm(obj_pos - GOAL) > 0.06:
        phase = np.array([0.0, 1.0, 0.0], dtype=np.float32)
    else:
        phase = np.array([0.0, 0.0, 1.0], dtype=np.float32)
    return np.concatenate([j_pos, j_vel, rel, obj_pos, ee_pos, ee_quat, goal_rel, phase]).astype(np.float32)


def main():
    world = World(stage_units_in_meters=1.0, physics_dt=1.0/120.0, rendering_dt=1.0/30.0)
    world.scene.add_default_ground_plane()
    franka = world.scene.add(Franka(prim_path="/World/Fancy_Franka", name="fancy_franka"))

    rx, ry = np.random.uniform(*OBJ_X_RANGE), np.random.uniform(*OBJ_Y_RANGE)
    u_disk = world.scene.add(DynamicCuboid(
        prim_path="/World/u_disk", name="u_disk",
        position=np.array([rx, ry, 0.02]),
        scale=np.array([0.05, 0.02, 0.01]),
        color=np.array([1.0, 0.0, 0.0]), mass=0.02,
    ))

    world.reset()
    franka.set_joint_positions(INIT_JOINTS)
    franka.apply_action(ArticulationAction(joint_positions=INIT_JOINTS))
    # 等几步让物理引擎稳定
    for _ in range(10):
        world.step(render=True)

    ik_solver = KinematicsSolver(franka)

    # 获取实际的 ee 朝下姿态（从初始位姿读取，比硬编码更可靠）
    _, ee_quat_init = franka.end_effector.get_world_pose()
    EE_QUAT_ACTUAL = np.array(ee_quat_init, dtype=np.float32).reshape(-1)[:4]
    print(f"EE orientation (from robot): {np.round(EE_QUAT_ACTUAL, 4)}")

    os.makedirs(os.path.dirname(args_cli.output), exist_ok=True)
    f = h5py.File(args_cli.output, "w")
    data_grp = f.create_group("data")
    data_grp.attrs["env_args"] = json.dumps({"env_name": "Isaac-UDisk-Grasp-v0", "env_kwargs": {}})

    demo_idx = 0
    ep_obs, ep_actions = [], []
    step = 0
    phase = "approach"
    close_counter = 0

    print(f"Collecting {args_cli.n_demos} demos -> {args_cli.output}")

    def step_toward(current, target, max_step=CARTESIAN_STEP):
        """笛卡尔空间增量移动，每步最多 max_step 米"""
        diff = target - current
        dist = np.linalg.norm(diff)
        if dist <= max_step:
            return target.copy()
        return current + diff / dist * max_step

    while simulation_app.is_running() and demo_idx < args_cli.n_demos:
        world.step(render=True)
        if not world.is_playing():
            continue

        step += 1
        j_pos = np.array(franka.get_joint_positions(), dtype=np.float32).reshape(-1)[:9]
        j_vel = np.array(franka.get_joint_velocities(), dtype=np.float32).reshape(-1)[:9]
        ee_pos, ee_quat_cur = franka.end_effector.get_world_pose()
        ee_pos = np.array(ee_pos, dtype=np.float32).reshape(-1)[:3]
        ee_quat_cur = np.array(ee_quat_cur, dtype=np.float32).reshape(-1)[:4]
        obj_pos, _ = u_disk.get_world_pose()
        obj_pos = np.array(obj_pos, dtype=np.float32).reshape(-1)[:3]

        obs = build_obs(j_pos, j_vel, ee_pos, ee_quat_cur, obj_pos)
        rel = obj_pos - ee_pos

        # === IK 控制 ===
        grip_val = 0.04  # 默认张开

        def apply_ik(target_pos, grip):
            """计算 IK 并应用（带关节 slew rate 限制），返回 9-dim action"""
            ik_act, success = ik_solver.compute_inverse_kinematics(
                target_position=target_pos, target_orientation=EE_QUAT_ACTUAL)
            if success:
                arm_joints = np.array(ik_act.joint_positions, dtype=np.float32).flatten()[:7]
                delta = arm_joints - j_pos[:7]
                delta = np.clip(delta, -JOINT_SLEW, JOINT_SLEW)
                arm_joints = j_pos[:7] + delta
                if step <= 5:
                    print(f"  IK ok: target={np.round(target_pos,3)} arm={np.round(arm_joints,3)} grip={grip}")
                full = np.zeros(9, dtype=np.float32)
                full[:7] = arm_joints
                full[7] = grip
                full[8] = grip
                franka.apply_action(ArticulationAction(joint_positions=full))
                return full.copy()
            else:
                if step <= 10:
                    print(f"  IK FAILED for target={np.round(target_pos,3)} ee={np.round(ee_pos,3)}")
                # IK 失败时保持当前位置
                full = j_pos.copy()
                full[7] = grip
                full[8] = grip
                franka.apply_action(ArticulationAction(joint_positions=full))
                return full

        if phase == "approach":
            final_target = obj_pos + np.array([0.0, 0.0, APPROACH_Z_OFFSET])
            action_record = apply_ik(step_toward(ee_pos, final_target), grip_val)
            if np.linalg.norm(ee_pos - final_target) < APPROACH_THRESH:
                phase = "descend"

        elif phase == "descend":
            final_target = obj_pos + np.array([0.0, 0.0, GRASP_Z_OFFSET])
            action_record = apply_ik(step_toward(ee_pos, final_target), grip_val)
            if np.linalg.norm(ee_pos - final_target) < DESCEND_THRESH:
                phase = "close"
                close_counter = 0

        elif phase == "close":
            grip_val = 0.0
            action_record = j_pos.copy()
            action_record[7:9] = grip_val
            franka.apply_action(ArticulationAction(joint_positions=action_record))
            close_counter += 1
            if close_counter >= CLOSE_WAIT:
                phase = "lift"

        elif phase == "lift":
            grip_val = 0.0
            final_target = np.array([obj_pos[0], obj_pos[1], GOAL[2]])
            action_record = apply_ik(step_toward(ee_pos, final_target), grip_val)

        should_record = (step % RECORD_STRIDE == 0) or phase in ("descend", "close")
        if should_record:
            ep_obs.append(obs)
            ep_actions.append(action_record.copy())

        if step % 60 == 0:
            print(f"  [demo {demo_idx}] step={step} phase={phase} rel={np.round(rel,3)} obj_z={obj_pos[2]:.3f} grip={j_pos[7]:.3f}")

        # === 成功检测 ===
        if phase == "lift" and obj_pos[2] > 0.15:
            if len(ep_actions) >= MIN_FRAMES:
                _save(data_grp, demo_idx, ep_obs, ep_actions)
                demo_idx += 1
                print(f"Demo {demo_idx}/{args_cli.n_demos} saved ({len(ep_actions)} frames)")
            _reset(franka, u_disk, world)
            ep_obs, ep_actions = [], []
            step, phase, close_counter = 0, "approach", 0
            continue

        # === 超时 ===
        if step >= MAX_STEPS:
            print(f"  Timeout (phase={phase}, obj_z={obj_pos[2]:.3f}), resetting")
            _reset(franka, u_disk, world)
            ep_obs, ep_actions = [], []
            step, phase, close_counter = 0, "approach", 0

    f.close()
    print(f"\nDone. {demo_idx} demos -> {args_cli.output}")


def _save(data_grp, idx, obs_list, act_list):
    grp = data_grp.create_group(f"demo_{idx}")
    grp.create_group("obs").create_dataset("policy", data=np.array(obs_list, dtype=np.float32))
    acts = np.nan_to_num(np.array(act_list, dtype=np.float32), nan=0.0)
    grp.create_dataset("actions", data=acts)
    dones = np.zeros(len(acts), dtype=np.float32)
    dones[-1] = 1.0
    grp.create_dataset("rewards", data=np.zeros(len(acts), dtype=np.float32))
    grp.create_dataset("dones", data=dones)
    grp.attrs["num_samples"] = len(acts)
    data_grp.attrs["total"] = idx + 1


def _reset(franka, u_disk, world):
    rx, ry = np.random.uniform(*OBJ_X_RANGE), np.random.uniform(*OBJ_Y_RANGE)
    u_disk.set_world_pose(position=np.array([rx, ry, 0.02]))
    franka.set_joint_positions(INIT_JOINTS)
    franka.apply_action(ArticulationAction(joint_positions=INIT_JOINTS))
    for _ in range(5):
        world.step(render=True)


if __name__ == "__main__":
    main()
    simulation_app.close()
