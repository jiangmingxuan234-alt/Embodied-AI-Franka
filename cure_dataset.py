import h5py
import numpy as np

file_path = "logs/robomimic/rmpflow_expert.hdf5"
lookahead = 15  # 🌟 核心猛药：强制让网络预测未来 15 帧的动作

print(f"💊 正在给数据集注射特效药：动作前瞻 (Lookahead={lookahead})...")

try:
    with h5py.File(file_path, "a") as f:
        data_grp = f["data"]
        for demo in data_grp.keys():
            actions = data_grp[demo]["actions"][:]
            
            # 动作往前平移 lookahead 帧
            shifted_actions = np.copy(actions)
            if len(actions) > lookahead:
                shifted_actions[:-lookahead] = actions[lookahead:]
                # 最后的几帧保持最后一个动作（紧紧抓住 U 盘）
                shifted_actions[-lookahead:] = actions[-1]
            
            # 覆盖原来的偷懒动作
            data_grp[demo]["actions"][...] = shifted_actions
            
    print("✅ 治疗完成！答案已被移走，网络必须自己学会走向 U 盘！")
except Exception as e:
    print(f"❌ 治疗失败: {e}")