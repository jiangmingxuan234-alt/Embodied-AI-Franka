from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlPpoActorCriticCfg, RslRlPpoAlgorithmCfg

def udisk_grasp_ppo_cfg() -> RslRlOnPolicyRunnerCfg:
    """机械臂抓取 U 盘任务的 PPO 超参数配置"""
    return RslRlOnPolicyRunnerCfg(
        seed=42,
        # 每次策略更新前，每个环境收集几步数据
        num_steps_per_env=24, 
        # 总共训练多少轮（抓取任务通常需要 1500~3000 轮才能收敛）
        max_iterations=1500,
        # 每隔多少轮保存一次模型权重
        save_interval=50,
        experiment_name="Franka_UDisk_Grasp",
        # 是否对观测值进行经验归一化（推荐开启，有助于稳定收敛）
        empirical_normalization=True,
        # 神经网络架构定义
        policy=RslRlPpoActorCriticCfg(
            init_noise_std=1.0,
            # Actor (策略网络)：决定机械臂怎么动
            actor_hidden_dims=[256, 128, 64],
            # Critic (价值网络)：评价当前状态好不好
            critic_hidden_dims=[256, 128, 64],
            activation="elu",
        ),
        # PPO 算法核心超参数
        algorithm=RslRlPpoAlgorithmCfg(
            value_loss_coef=1.0,
            use_clipped_value_loss=True,
            clip_param=0.2,
            entropy_coef=0.006, # 鼓励探索
            num_learning_epochs=5,
            num_mini_batches=4,
            learning_rate=1e-3, # 初始学习率
            schedule="adaptive", # 自适应学习率衰减
            gamma=0.99,
            lam=0.95,
            desired_kl=0.01,
            max_grad_norm=1.0,
        ),
    )