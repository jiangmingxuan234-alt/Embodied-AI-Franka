"""
数据后处理：在夹爪闭合前插入下压帧
原始数据中 ee 在物体上方 ~5.4cm 就闭合，需要补充下压到 ~1.5cm 的数据。

方法：找到每个 demo 中夹爪闭合的帧，取闭合前最后几帧的关节运动趋势，
沿该趋势外推生成继续下压的帧，然后接上闭合帧。
"""
import h5py
import numpy as np

INPUT = "/home/jmx001/my_program/my_robot_project/logs/robomimic/full_grasp_abs_v5.hdf5"
OUTPUT = "/home/jmx001/my_program/my_robot_project/logs/robomimic/full_grasp_abs_v6.hdf5"

N_INSERT = 25  # 插入帧数

fi = h5py.File(INPUT, "r")
fo = h5py.File(OUTPUT, "w")
data_out = fo.create_group("data")
data_out.attrs["env_args"] = fi["data"].attrs["env_args"]

n_demos = len(fi["data"].keys())
augmented = 0

for di in range(n_demos):
    key = f"demo_{di}"
    obs = fi["data"][key]["obs"]["policy"][:]
    act = fi["data"][key]["actions"][:]
    n = min(len(obs), len(act))
    obs, act = obs[:n], act[:n]

    # 找夹爪闭合帧（action 中 grip < 0.02）
    close_idx = None
    for i in range(n):
        if act[i, 7] < 0.02:
            close_idx = i
            break

    if close_idx is None or close_idx < 10:
        grp = data_out.create_group(key)
        grp.create_group("obs").create_dataset("policy", data=obs)
        grp.create_dataset("actions", data=act)
        dones = np.zeros(n, dtype=np.float32); dones[-1] = 1.0
        grp.create_dataset("dones", data=dones)
        grp.create_dataset("rewards", data=np.zeros(n, dtype=np.float32))
        grp.attrs["num_samples"] = n
        continue

    # 取闭合前 5 帧的关节速度趋势（action 的差分）
    window = 5
    start = max(0, close_idx - window)
    joint_vel = (act[close_idx-1, :7] - act[start, :7]) / (close_idx - 1 - start)

    # 生成插入帧：沿趋势外推，夹爪保持张开
    insert_obs = []
    insert_act = []
    base_obs = obs[close_idx - 1].copy()
    base_act = act[close_idx - 1].copy()

    for k in range(N_INSERT):
        new_act = base_act.copy()
        new_act[:7] = base_act[:7] + joint_vel * (k + 1)
        # 前 N_INSERT-3 帧夹爪张开，最后 3 帧闭合
        if k >= N_INSERT - 3:
            new_act[7] = 0.0
            new_act[8] = 0.0
        else:
            new_act[7] = 0.04
            new_act[8] = 0.04

        # obs: 复制 base，更新 j_pos 和 grip
        new_obs = base_obs.copy()
        new_obs[:7] = new_act[:7]
        new_obs[7] = new_act[7]
        new_obs[8] = new_act[8]
        # 近似更新 ee_z（每帧下降约 dz）
        # 从数据看 joint_vel 对应的 ee 下降速率约 1mm/frame
        dz_per_frame = 0.0015  # 1.5mm per frame
        new_obs[26] -= dz_per_frame * (k + 1)  # ee_z
        new_obs[20] = new_obs[23] - new_obs[26]  # rel_z = obj_z - ee_z

        insert_obs.append(new_obs)
        insert_act.append(new_act)

    new_obs_arr = np.concatenate([obs[:close_idx], np.array(insert_obs), obs[close_idx:]], axis=0)
    new_act_arr = np.concatenate([act[:close_idx], np.array(insert_act), act[close_idx:]], axis=0)

    grp = data_out.create_group(key)
    grp.create_group("obs").create_dataset("policy", data=new_obs_arr.astype(np.float32))
    grp.create_dataset("actions", data=new_act_arr.astype(np.float32))
    nn = len(new_act_arr)
    dones = np.zeros(nn, dtype=np.float32); dones[-1] = 1.0
    grp.create_dataset("dones", data=dones)
    grp.create_dataset("rewards", data=np.zeros(nn, dtype=np.float32))
    grp.attrs["num_samples"] = nn
    augmented += 1

    if di < 3:
        print(f"  {key}: {n} -> {nn} frames (+{N_INSERT})")

data_out.attrs["total"] = n_demos
fi.close()
fo.close()
print(f"Done. Augmented {augmented}/{n_demos} demos -> {OUTPUT}")
