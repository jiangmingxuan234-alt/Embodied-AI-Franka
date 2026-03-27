"""
数据清洗：截断悬停区重复帧 + 修正 grip action
输入: full_grasp_abs_v2.hdf5
输出: full_grasp_clean.hdf5

问题:
1. 50% 帧在 rel_z ∈ [-0.08, -0.05] 悬停区，action 几乎不变 → 模型学到"停住"
2. grip action 有负值 (-0.04)，被 clip 后信号丢失

修复:
1. 对每个 demo，在悬停区只保留每 N 帧一帧（下采样）
2. grip action clip 到 [0, 0.04]，闭合用 0.0，张开用 0.04
"""
import h5py
import numpy as np
import json
import os

INPUT = "/home/jmx001/my_program/my_robot_project/logs/robomimic/full_grasp_abs_v2.hdf5"
OUTPUT = "/home/jmx001/my_program/my_robot_project/logs/robomimic/full_grasp_clean.hdf5"

# 悬停区: rel_z ∈ [-0.08, -0.05]，只保留每 HOVER_STRIDE 帧
HOVER_STRIDE = 8


def clean_demo(obs, act):
    n = min(len(obs), len(act))
    obs, act = obs[:n], act[:n]

    # 修正 grip action: clip 到 [0, 0.04]
    act[:, 7] = np.clip(act[:, 7], 0.0, 0.04)
    act[:, 8] = np.clip(act[:, 8], 0.0, 0.04)

    # 找到 grip 闭合帧
    grip_obs = obs[:, 7]
    close_idx = -1
    for i in range(1, n):
        if grip_obs[i] < 0.035 and grip_obs[i-1] >= 0.035:
            close_idx = i
            break

    # 下采样悬停区
    keep = []
    hover_count = 0
    for i in range(n):
        rel_z = obs[i, 20]
        in_hover = -0.08 < rel_z < -0.05

        if in_hover and (close_idx < 0 or i < close_idx):
            # 悬停区 + 还没闭合 → 下采样
            hover_count += 1
            if hover_count % HOVER_STRIDE == 1:  # 保留第 1, 9, 17... 帧
                keep.append(i)
        else:
            keep.append(i)
            hover_count = 0

    return obs[keep], act[keep]


def main():
    fin = h5py.File(INPUT, "r")
    demos = sorted(fin["data"].keys(), key=lambda x: int(x.split("_")[1]))
    print(f"Input: {len(demos)} demos")

    os.makedirs(os.path.dirname(OUTPUT), exist_ok=True)
    fout = h5py.File(OUTPUT, "w")
    dg = fout.create_group("data")
    if "env_args" in fin["data"].attrs:
        dg.attrs["env_args"] = fin["data"].attrs["env_args"]

    total_before, total_after = 0, 0
    for idx, dk in enumerate(demos):
        obs = fin["data"][dk]["obs"]["policy"][:]
        act = fin["data"][dk]["actions"][:]
        total_before += min(len(obs), len(act))

        obs_c, act_c = clean_demo(obs, act)
        total_after += len(obs_c)

        grp = dg.create_group(f"demo_{idx}")
        grp.create_group("obs").create_dataset("policy", data=obs_c.astype(np.float32))
        grp.create_dataset("actions", data=act_c.astype(np.float32))
        dones = np.zeros(len(act_c), dtype=np.float32)
        dones[-1] = 1.0
        grp.create_dataset("rewards", data=np.zeros(len(act_c), dtype=np.float32))
        grp.create_dataset("dones", data=dones)
        grp.attrs["num_samples"] = len(act_c)

    dg.attrs["total"] = len(demos)
    fin.close()
    fout.close()

    print(f"Output: {OUTPUT}")
    print(f"Frames: {total_before} → {total_after} ({total_after/total_before*100:.1f}%)")

    # 验证分布
    f = h5py.File(OUTPUT, "r")
    all_rz = []
    for dk in f["data"].keys():
        all_rz.extend(f["data"][dk]["obs"]["policy"][:, 20].tolist())
    rz = np.array(all_rz)
    bins = [(-0.5,-0.3),(-0.3,-0.2),(-0.2,-0.15),(-0.15,-0.1),(-0.1,-0.08),(-0.08,-0.05),(-0.05,-0.02),(-0.02,0.5)]
    print("\nrel_z distribution after cleaning:")
    for lo, hi in bins:
        m = (rz >= lo) & (rz < hi)
        print(f"  [{lo:+.2f}, {hi:+.2f}): {m.sum():6d} ({m.sum()/len(rz)*100:5.1f}%)")
    f.close()


if __name__ == "__main__":
    main()
