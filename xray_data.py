import h5py
import numpy as np

file_path = "logs/robomimic/rmpflow_expert.hdf5"
print(f"🔍 正在对 1000 条数据集进行深度 X 光扫描: {file_path}")

try:
    with h5py.File(file_path, "r") as f:
        data_grp = f["data"]
        demos = list(data_grp.keys())
        
        total_steps = 0
        frozen_steps = 0
        all_obs = []
        
        # 随机抽查前 100 个 Demo 就足够看出问题了
        check_num = min(100, len(demos))
        
        for demo_name in demos[:check_num]:
            obs = data_grp[demo_name]["obs"]["policy"][:]
            actions = data_grp[demo_name]["actions"][:]
            
            total_steps += len(actions)
            all_obs.append(obs)
            
            # 检查这 100 个 Demo 里，有多少步是“原地发呆”的
            # 如果前后两帧的动作差异极小（小于1e-4），我们认为它卡住了
            action_diff = np.abs(np.diff(actions, axis=0))
            frozen_steps += np.sum(np.max(action_diff, axis=1) < 1e-4)

        print("-" * 50)
        print(f"📊 【数据质量报告】 (抽查了 {check_num} 条轨迹)")
        print(f"总步数: {total_steps}")
        print(f"原地发呆步数: {frozen_steps}")
        frozen_ratio = (frozen_steps / total_steps) * 100
        print(f"⚠️ 发呆比例: {frozen_ratio:.2f}%")
        
        if frozen_ratio > 20.0:
            print("❌ 结论：专家数据已废！机器人大部分时间都在卡顿/发呆，难怪学不会！")
        else:
            print("✅ 结论：动作很连贯，专家数据质量没问题！")
            
        print("-" * 50)
        print("📊 【归一化致命弱点排查】")
        all_obs = np.concatenate(all_obs, axis=0)
        obs_variance = np.var(all_obs, axis=0)
        
        zero_var_dims = np.where(obs_variance < 1e-8)[0]
        if len(zero_var_dims) > 0:
            print(f"💥 发现方差为 0 的维度 (罪魁祸首): 第 {zero_var_dims} 维")
            print("❌ 结论：这些维度在 1000 条数据里永远没变化！一开自动归一化，网络必定除以 0 当场暴毙！")
        else:
            print("✅ 观测数据方差健康，可以直接开启归一化！")
            
except Exception as e:
    print(f"扫描失败: {e}")