"""
VLA 数据采集 — 带腕部相机的完整抓取数据
在 collect_full_grasp.py 基础上增加 128x128 RGB 图像采集。
"""
import argparse
from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
AppLauncher.add_app_launcher_args(parser)
parser.add_argument("--n_demos", type=int, default=300)
parser.add_argument("--output", type=str,
    default="/home/jmx001/my_program/my_robot_project/logs/robomimic/full_grasp_vision.hdf5")
parser.set_defaults(enable_cameras=True)
args_cli = parser.parse_args()
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import omni.kit.app
ext_manager = omni.kit.app.get_app().get_extension_manager()
ext_manager.set_extension_enabled_immediate("omni.isaac.core", True)
ext_manager.set_extension_enabled_immediate("omni.isaac.franka", True)
ext_manager.set_extension_enabled_immediate("omni.isaac.sensor", True)

import numpy as np
import h5py
import json
import os

from omni.isaac.core import World
from omni.isaac.franka import Franka
from omni.isaac.core.objects import DynamicCuboid
from omni.isaac.franka.controllers import PickPlaceController
from omni.isaac.core.utils.types import ArticulationAction
from omni.isaac.sensor import Camera

OBJ_X_RANGE = (0.42, 0.58)
OBJ_Y_RANGE = (-0.30, 0.30)
GOAL = np.array([0.40, 0.0, 0.40], dtype=np.float32)
INIT_JOINTS = np.array([0.0, -1.1, 0.0, -2.3, 0.0, 2.4, 0.8, 0.04, 0.04], dtype=np.float32)
MAX_STEPS = 1500
MIN_FRAMES = 100
IMG_SIZE = 128


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

    # 第三人称俯视相机（固定位置，看整个工作台）
    camera = Camera(
        prim_path="/World/overhead_cam",
        resolution=(IMG_SIZE, IMG_SIZE),
        frequency=30,
    )
    world.scene.add(camera)

    world.reset()
    franka.set_joint_positions(INIT_JOINTS)
    franka.apply_action(ArticulationAction(joint_positions=INIT_JOINTS))

    camera.set_world_pose(
        position=np.array([0.5, 0.0, 0.8]),
        orientation=np.array([0.0, 0.707, 0.0, 0.707]),
    )
    camera.initialize()

    ctrl = PickPlaceController(name="ctrl", gripper=franka.gripper, robot_articulation=franka)

    os.makedirs(os.path.dirname(args_cli.output), exist_ok=True)
    f = h5py.File(args_cli.output, "w")
    data_grp = f.create_group("data")
    data_grp.attrs["env_args"] = json.dumps({"env_name": "Isaac-UDisk-Grasp-VLA-v0", "env_kwargs": {}})

    demo_idx = 0
    ep_obs, ep_actions, ep_images = [], [], []
    step = 0
    last_valid_action = np.array(INIT_JOINTS, dtype=np.float32)
    post_frames = 0

    # 预热相机（前几帧可能返回 None）
    for _ in range(10):
        world.step(render=True)

    print(f"Collecting {args_cli.n_demos} demos with {IMG_SIZE}x{IMG_SIZE} images -> {args_cli.output}")

    while simulation_app.is_running() and demo_idx < args_cli.n_demos:
        world.step(render=True)
        if not world.is_playing():
            continue

        step += 1
        j_pos = np.array(franka.get_joint_positions(), dtype=np.float32).reshape(-1)[:9]
        j_vel = np.array(franka.get_joint_velocities(), dtype=np.float32).reshape(-1)[:9]
        ee_pos, ee_quat = franka.end_effector.get_world_pose()
        ee_pos = np.array(ee_pos, dtype=np.float32).reshape(-1)[:3]
        ee_quat = np.array(ee_quat, dtype=np.float32).reshape(-1)[:4]
        obj_pos, _ = u_disk.get_world_pose()
        obj_pos = np.array(obj_pos, dtype=np.float32).reshape(-1)[:3]

        obs = build_obs(j_pos, j_vel, ee_pos, ee_quat, obj_pos)

        # 采集图像
        rgba = camera.get_rgba()
        if rgba is None or rgba.shape[0] != IMG_SIZE:
            continue
        rgb = rgba[:, :, :3].astype(np.uint8)

        safe_pick = np.clip(obj_pos, a_min=[0.2, -0.5, 0.0], a_max=[0.8, 0.5, 0.6])
        actions = ctrl.forward(
            picking_position=safe_pick,
            placing_position=GOAL,
            current_joint_positions=j_pos,
        )
        franka.apply_action(actions)

        action_array = np.array(j_pos, dtype=np.float32)
        if actions.joint_positions is not None:
            l = len(actions.joint_positions)
            if actions.joint_indices is not None:
                action_array[actions.joint_indices] = actions.joint_positions
            else:
                action_array[:l] = actions.joint_positions

        nan_mask = np.isnan(action_array) | np.isinf(action_array)
        action_array[nan_mask] = last_valid_action[nan_mask]
        last_valid_action = action_array.copy()

        if np.linalg.norm(ee_pos) > 2.0:
            _reset(franka, u_disk, ctrl)
            ep_obs, ep_actions, ep_images = [], [], []
            step, post_frames = 0, 0
            last_valid_action = np.array(INIT_JOINTS, dtype=np.float32)
            continue

        # 记录
        rel = obj_pos - ee_pos
        dist_xy = float(np.linalg.norm(rel[:2]))
        grasp_critical = j_pos[7] < 0.035 or obj_pos[2] > 0.05
        near_object = rel[2] > -0.15 and dist_xy < 0.15
        mid_zone = rel[2] > -0.30 and rel[2] <= -0.15
        if grasp_critical or near_object:
            record = True
        elif mid_zone and step % 4 == 0:
            record = True
        elif step % 8 == 0:
            record = True
        else:
            record = False
        if record:
            ep_obs.append(obs)
            ep_actions.append(action_array.copy())
            ep_images.append(rgb)

        if step % 60 == 0:
            print(f"  [demo {demo_idx}] step={step} rel={np.round(rel,3)} obj_z={obj_pos[2]:.3f} frames={len(ep_obs)}")

        # 成功检测
        lifted = obj_pos[2] > 0.09
        at_goal = np.linalg.norm(obj_pos - GOAL) < 0.08
        if lifted and at_goal and post_frames == 0:
            post_frames = 1
        if post_frames > 0:
            post_frames += 1
            if post_frames > 10:
                if len(ep_actions) >= MIN_FRAMES:
                    _save(data_grp, demo_idx, ep_obs, ep_actions, ep_images)
                    demo_idx += 1
                    print(f"Demo {demo_idx}/{args_cli.n_demos} saved ({len(ep_actions)} frames, with images)")
                _reset(franka, u_disk, ctrl)
                ep_obs, ep_actions, ep_images = [], [], []
                step, post_frames = 0, 0
                last_valid_action = np.array(INIT_JOINTS, dtype=np.float32)
                continue

        if step >= MAX_STEPS:
            _reset(franka, u_disk, ctrl)
            ep_obs, ep_actions, ep_images = [], [], []
            step, post_frames = 0, 0
            last_valid_action = np.array(INIT_JOINTS, dtype=np.float32)

    f.close()
    print(f"\nDone. {demo_idx} demos -> {args_cli.output}")


def _save(data_grp, idx, obs_list, act_list, img_list):
    grp = data_grp.create_group(f"demo_{idx}")
    obs_grp = grp.create_group("obs")
    obs_grp.create_dataset("policy", data=np.array(obs_list, dtype=np.float32))
    obs_grp.create_dataset("images", data=np.array(img_list, dtype=np.uint8),
                           compression="gzip", compression_opts=4)
    acts = np.nan_to_num(np.array(act_list, dtype=np.float32), nan=0.0)
    grp.create_dataset("actions", data=acts)
    dones = np.zeros(len(acts), dtype=np.float32)
    dones[-1] = 1.0
    grp.create_dataset("rewards", data=np.zeros(len(acts), dtype=np.float32))
    grp.create_dataset("dones", data=dones)
    grp.attrs["num_samples"] = len(acts)
    data_grp.attrs["total"] = idx + 1


def _reset(franka, u_disk, ctrl):
    rx, ry = np.random.uniform(*OBJ_X_RANGE), np.random.uniform(*OBJ_Y_RANGE)
    u_disk.set_world_pose(position=np.array([rx, ry, 0.02]))
    franka.set_joint_positions(INIT_JOINTS)
    franka.apply_action(ArticulationAction(joint_positions=INIT_JOINTS))
    ctrl.reset()


if __name__ == "__main__":
    main()
    simulation_app.close()
