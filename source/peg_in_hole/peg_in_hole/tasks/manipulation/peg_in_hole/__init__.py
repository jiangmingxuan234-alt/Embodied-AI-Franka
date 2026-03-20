import gymnasium as gym

gym.register(
    id="Isaac-UDisk-Grasp-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": f"{__name__}.env_cfg:GraspEnvCfg", 
        # 🌟 新增：告诉官方训练脚本，我的 PPO 配置在这个函数里！
        "rsl_rl_cfg_entry_point": f"{__name__}.agents:udisk_grasp_ppo_cfg", 
    },
    disable_env_checker=True,
)