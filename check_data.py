import h5py
import numpy as np

file_path = "logs/robomimic/rmpflow_expert.hdf5"
print(f"🔍 正在扫描绝世武功秘籍: {file_path}")

try:
    with h5py.File(file_path, "r") as f:
        data_grp = f["data"]
        nan_count = 0
        inf_count = 0
        
        for demo_name in data_grp.keys():
            obs = data_grp[demo_name]["obs"]["policy"][:]
            actions = data_grp[demo_name]["actions"][:]
            
            # 检查是否有 NaN (非数字)
            if np.isnan(obs).any() or np.isnan(actions).any():
                print(f"🚨 抓到内鬼：在 {demo_name} 中发现了 NaN！")
                nan_count += 1
                
            # 检查是否有 Inf (无穷大)
            if np.isinf(obs).any() or np.isinf(actions).any():
                print(f"🚨 抓到内鬼：在 {demo_name} 中发现了 Inf (无穷大)！")
                inf_count += 1
                
        print("-" * 40)
        if nan_count == 0 and inf_count == 0:
            print("✅ 数据集非常健康，没有任何 NaN 或 Inf！(那事情就更诡异了...)")
        else:
            print(f"💥 破案了！有 {nan_count} 个 Demo 包含 NaN，{inf_count} 个包含 Inf。")
            print("💀 网络吃了一口毒数据，当场暴毙！难怪怎么调参都没用！")
            
except Exception as e:
    print(f"❌ 扫描失败: {e}")