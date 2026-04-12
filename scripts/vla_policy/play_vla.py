"""
VLA 验证脚本 — Action Chunking + Temporal Ensemble
"""
import argparse
from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
AppLauncher.add_app_launcher_args(parser)
parser.add_argument("--checkpoint", type=str,
    default="/home/jmx001/my_program/my_robot_project/logs/vla_policy/vla_chunk_epoch_300.pt")
parser.add_argument("--norm_stats", type=str,
    default="/home/jmx001/my_program/my_robot_project/logs/vla_policy/norm_stats.npz")
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
import torch
import torchvision.transforms as T

from omni.isaac.core import World
from omni.isaac.franka import Franka
from omni.isaac.core.objects import DynamicCuboid
from omni.isaac.core.utils.types import ArticulationAction
from omni.isaac.sensor import Camera

import sys
sys.path.insert(0, "/home/jmx001/my_program/my_robot_project/scripts/vla_policy")
from train_vla import VLAPolicy, SimpleTokenizer

GOAL = np.array([0.40, 0.0, 0.40], dtype=np.float32)
INIT_JOINTS = np.array([0.0, -1.1, 0.0, -2.3, 0.0, 2.4, 0.8, 0.04, 0.04], dtype=np.float32)
J_LOW = np.array([-2.8973, -1.7628, -2.8973, -3.0718, -2.8973, -0.0175, -2.8973, 0.0, 0.0])
J_HIGH = np.array([2.8973, 1.7628, 2.8973, -0.0698, 2.8973, 3.7525, 2.8973, 0.04, 0.04])
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
TEMPORAL_M = 0.1


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
    ckpt = torch.load(args_cli.checkpoint, map_location=DEVICE, weights_only=False)
    cfg = ckpt["config"]
    instruction = cfg.get("instruction", "pick up the red USB disk")
    chunk_size = cfg.get("chunk_size", 16)

    tokenizer = SimpleTokenizer()
    tokenizer.fit(instruction)
    tokens = torch.tensor(tokenizer.encode(instruction), dtype=torch.long).unsqueeze(0).to(DEVICE)

    model = VLAPolicy(
        obs_dim=cfg["obs_dim"], action_dim=cfg["action_dim"],
        chunk_size=chunk_size, vocab_size=cfg.get("vocab_size", 64),
    ).to(DEVICE)
    model.load_state_dict(ckpt["model"], strict=False)
    model.load_vit()
    model.vit = model.vit.to(DEVICE)
    model.eval()

    stats = np.load(args_cli.norm_stats)
    act_min, act_max = stats["act_min"], stats["act_max"]
    obs_min, obs_max = stats["obs_min"], stats["obs_max"]
    act_range = act_max - act_min
    act_range[act_range < 1e-6] = 1.0
    obs_range = obs_max - obs_min
    obs_range[obs_range < 1e-6] = 1.0

    img_transform = T.Compose([
        T.ToPILImage(),
        T.Resize((224, 224)),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    world = World(stage_units_in_meters=1.0, physics_dt=1.0/120.0, rendering_dt=1.0/30.0)
    world.scene.add_default_ground_plane()
    franka = world.scene.add(Franka(prim_path="/World/Fancy_Franka", name="fancy_franka"))
    u_disk = world.scene.add(DynamicCuboid(
        prim_path="/World/u_disk", name="u_disk",
        position=np.array([0.5, 0.0, 0.02]),
        scale=np.array([0.05, 0.02, 0.01]),
        color=np.array([1.0, 0.0, 0.0]), mass=0.02,
    ))
    camera = Camera(prim_path="/World/overhead_cam", resolution=(128, 128), frequency=30)
    world.scene.add(camera)

    world.reset()
    franka.set_joint_positions(INIT_JOINTS)
    franka.apply_action(ArticulationAction(joint_positions=INIT_JOINTS))
    camera.set_world_pose(
        position=np.array([0.5, 0.0, 0.8]),
        orientation=np.array([0.0, 0.707, 0.0, 0.707]),
    )
    camera.initialize()

    for _ in range(10):
        world.step(render=True)

    action_chunks = []  # list of (chunk_array, start_step)
    step = 0
    prev_dist = None
    stall_count = 0
    descend_active = False
    descend_step = 0

    # 从数据集提取的"抓取位置"平均关节配置
    GRASP_JOINTS = np.array([-0.0154, 0.5611, -0.0057, -2.242, 0.0076, 2.8017, 0.7585, 0.0, 0.0], dtype=np.float32)

    print(f"VLA Policy (chunk={chunk_size}): {args_cli.checkpoint}")
    print(f"Instruction: \"{instruction}\"")

    while simulation_app.is_running():
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

        rel = obj_pos - ee_pos
        dist = float(np.linalg.norm(rel))
        dist_xy = float(np.linalg.norm(rel[:2]))
        dist_z = float(abs(rel[2]))

        # 悬停检测
        if not descend_active:
            if prev_dist is not None and abs(prev_dist - dist) < 0.005:
                stall_count += 1
            else:
                stall_count = 0
            prev_dist = dist

            if stall_count > 15 and dist_xy < 0.10:
                descend_active = True
                descend_step = 0
                print(f"[VLA] step={step} DESCEND ACTIVATED ee_z={ee_pos[2]:.3f}")

        # DESCEND 模式：缓慢插值到数据集中的抓取关节配置
        if descend_active:
            descend_step += 1
            target = j_pos.copy()

            # 缓慢逼近抓取配置（每步 2% 插值）
            target[:7] = j_pos[:7] + 0.02 * (GRASP_JOINTS[:7] - j_pos[:7])

            if descend_step < 80:
                # 下压阶段：半闭合
                target[7] = 0.015
                target[8] = 0.015
                status = "descending"
            else:
                # 闭合并抬升
                target[7] = 0.0
                target[8] = 0.0
                LIFT_JOINTS = np.array([0.0, -0.5, 0.0, -2.0, 0.0, 2.4, 0.8], dtype=np.float32)
                target[:7] = j_pos[:7] + 0.01 * (LIFT_JOINTS - j_pos[:7])
                status = "GRASPING"

            target = np.clip(target, J_LOW, J_HIGH).astype(np.float32)
            franka.apply_action(ArticulationAction(joint_positions=target))

            if step % 15 == 0:
                print(f"[VLA] step={step} {status} ee_z={ee_pos[2]:.3f} obj_z={obj_pos[2]:.3f} d={descend_step}")
            continue

        rgba = camera.get_rgba()
        if rgba is None or rgba.shape[0] < 128:
            continue
        rgb = rgba[:, :, :3].astype(np.uint8)
        img_tensor = img_transform(rgb).unsqueeze(0).to(DEVICE)

        obs = build_obs(j_pos, j_vel, ee_pos, ee_quat, obj_pos)
        obs_norm = (2.0 * (obs - obs_min) / obs_range - 1.0).astype(np.float32)
        obs_tensor = torch.from_numpy(obs_norm).unsqueeze(0).to(DEVICE)

        with torch.no_grad():
            pred_norm = model.inference(img_tensor, obs_tensor, tokens)
        pred_np = pred_norm.squeeze(0).cpu().numpy()
        pred_actions = 0.5 * (pred_np + 1.0) * act_range + act_min
        action_chunks.append((pred_actions, step))

        weights = []
        actions = []
        for chunk, start in action_chunks:
            idx = step - start
            if 0 <= idx < len(chunk):
                weights.append(np.exp(-TEMPORAL_M * idx))
                actions.append(chunk[idx])

        target = np.average(actions, axis=0, weights=weights).astype(np.float32)

        grip_val = 0.04 if pred_actions[0, 7] > 0.02 else 0.0
        target[7] = grip_val
        target[8] = grip_val

        if dist_xy < 0.09 and dist_z < 0.12:
            target[7] = 0.0
            target[8] = 0.0

        action_chunks = [(c, s) for c, s in action_chunks if step - s < len(c)]

        target = np.clip(target, J_LOW, J_HIGH).astype(np.float32)
        franka.apply_action(ArticulationAction(joint_positions=target))

        if step % 15 == 0:
            n_chunks = len(action_chunks)
            print(f"[VLA] step={step} dist={np.linalg.norm(rel):.3f} rel={np.round(rel, 3)} "
                  f"obj_z={obj_pos[2]:.3f} chunks={n_chunks}")


if __name__ == "__main__":
    main()
    simulation_app.close()
