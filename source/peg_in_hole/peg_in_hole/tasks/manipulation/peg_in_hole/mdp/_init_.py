import gymnasium as gym
from . import env_cfg

# 注册我们的 6D 抓取环境
gym.register(
    id="Isaac-UDisk-Grasp-v0",             # 这是你以后训练时用的名字
    entry_point="isaaclab.envs:RLTaskEnv", # 入口点统一用官方的基类
    kwargs={
        "env_cfg_entry_point": f"{__name__}.env_cfg:GraspEnvCfg", # 绑定你刚才写的配置类
    },
    disable_env_checker=True,
)