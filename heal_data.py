import h5py
import numpy as np

file_path = "logs/robomimic/rmpflow_expert.hdf5"
print("🏥 正在给 1000 个 Demo 做透析手术，清理剧毒的 NaN...")

try:
    with h5py.File(file_path, "a") as f:  # 注意这里是 "a" 模式，可以直接修改文件
        data_grp = f["data"]
        fixed_action_count = 0
        fixed_obs_count = 0
        
        for demo_name in data_grp.keys():
            obs = data_grp[demo_name]["obs"]["policy"][:]
            actions = data_grp[demo_name]["actions"][:]
            
            # 💉 解毒 Actions：如果动作里有 NaN（RMPflow不想动的关节）
            # 我们就用当前那一帧的“实际关节位置”去替换它，强行告诉网络“保持在这里”
            action_mask = np.isnan(actions)
            if action_mask.any():
                # obs 的前 9 维就是当前的关节位置
                actions[action_mask] = obs[:, :9][action_mask]
                # 重新写入硬盘
                data_grp[demo_name]["actions"][...] = actions
                fixed_action_count += 1
                
            # 💉 解毒 Obs（以防万一有观测数据也是 NaN）
            obs_mask = np.isnan(obs)
            if obs_mask.any():
                obs[obs_mask] = 0.0 # 观测数据缺了就补 0
                data_grp[demo_name]["obs"]["policy"][...] = obs
                fixed_obs_count += 1
                
        print("-" * 40)
        print(f"✨ 手术大成功！")
        print(f"🧹 修复了 {fixed_action_count} 个带毒的 Actions 序列。")
        print(f"🧹 修复了 {fixed_obs_count} 个带毒的 Obs 序列。")
        print("✅ 数据集已彻底纯净，可以放心喂给神经网络了！")
            
except Exception as e:
    print(f"❌ 手术失败: {e}")