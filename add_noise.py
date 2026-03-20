import h5py
import numpy as np

file_path = "logs/robomimic/rmpflow_expert.hdf5"
print(f"💉 正在给数据集注入微量波动...")

try:
    with h5py.File(file_path, "a") as f:
        data_grp = f["data"]
        for demo in data_grp.keys():
            obs = data_grp[demo]["obs"]["policy"][:]
            
            # 核心黑科技：注入 0.00001 的极小噪音！
            # 物理引擎根本感觉不到，但数学上方差再也不是 0 了！归一化绝对不会再炸！
            noise = np.random.normal(0, 1e-5, obs.shape)
            data_grp[demo]["obs"]["policy"][...] = obs + noise
            
    print("✅ 疫苗注射成功！所有数据的方差已被激活，现在可以开启归一化了！")
except Exception as e:
    print(f"❌ 失败: {e}")