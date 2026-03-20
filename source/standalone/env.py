import math
import isaaclab.envs.mdp as mdp
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.managers import ObservationGroupCfg, ObservationTermCfg, ActionTermCfg, RewardTermCfg, SceneEntityCfg, EventTermCfg, TerminationTermCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.utils import configclass
from isaaclab.assets import ArticulationCfg, RigidObjectCfg
from isaaclab.sim import spawner as sim_utils

# 🌟 导入 Isaac Lab 官方内置的 Franka 模型配置
from isaaclab_assets.robots.franka import FRANKA_PANDA_CFG

# 🌟 导入我们之前写的自定义奖励函数
# 注意：这要求你的 rewards.py 必须放在当前文件夹下的 mdp 文件夹里
from .mdp import rewards as custom_rewards

@configclass
class GraspSceneCfg(InteractiveSceneCfg):
    # 1. 地面
    ground = mdp.GroundPlaneCfg()

    # 2. 机器人：使用官方配置，并挂载到环境路径下
    robot: ArticulationCfg = FRANKA_PANDA_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")

    # 3. U 盘：纯代码生成红色刚体方块
    u_disk = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/UDisk",
        spawn=sim_utils.CuboidCfg(
            size=[0.12, 0.03, 0.025], # 我们之前量好的绝佳尺寸
            rigid_props=sim_utils.RigidBodyPropertiesCfg(mass=0.02),
            mass_props=sim_utils.MassPropertiesCfg(density=400.0),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(1.0, 0.0, 0.0)), # 红色
            physics_material=sim_utils.RigidBodyMaterialCfg(static_friction=2.0, dynamic_friction=2.0) # 高摩擦力
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0.45, 0.0, 0.02)),
    )

    # 4. 场景光照
    light = sim_utils.DomeLightCfg(intensity=2000.0, color=(1.0, 1.0, 1.0))


@configclass
class ActionsCfg:
    # 控制大臂 7 个关节 + 夹爪 2 个关节
    arm_action = mdp.JointPositionActionCfg(
        asset_name="robot",
        joint_names=["panda_joint.*", "panda_finger.*"],
        scale=1.0,
        use_default_offset=True
    )


@configclass
class ObservationsCfg:
    @configclass
    class PolicyCfg(ObservationGroupCfg):
        # 自身感知：关节位置与速度
        joint_pos = ObservationTermCfg(func=mdp.joint_pos_rel, params={"asset_cfg": SceneEntityCfg("robot")})
        joint_vel = ObservationTermCfg(func=mdp.joint_vel_rel, params={"asset_cfg": SceneEntityCfg("robot")})
        
        # 目标感知：U 盘在机器人坐标系下的位姿
        object_pose = ObservationTermCfg(
            func=mdp.object_pose_in_robot_root_frame,
            params={"object_cfg": SceneEntityCfg("u_disk"), "robot_cfg": SceneEntityCfg("robot")}
        )
        
        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = True


@configclass
class EventCfg:
    # 🌟 核心机制：回合重置时，U 盘在桌面上随机出现并随机旋转
    randomize_object = EventTermCfg(
        func=mdp.reset_root_state_uniform,
        mode="reset",
        params={
            # XY 范围内随机位置，Yaw 轴 (Z轴) 旋转范围 -180度 到 180度
            "pose_range": {"x": (0.35, 0.55), "y": (-0.2, 0.2), "yaw": (-math.pi, math.pi)},
            "velocity_range": {},
            "asset_cfg": SceneEntityCfg("u_disk")
        }
    )
    
    # 回合重置时，机械臂恢复初始姿态
    reset_robot_joints = EventTermCfg(
        func=mdp.reset_joints_by_scale,
        mode="reset",
        params={
            "position_range": (0.0, 0.0), 
            "velocity_range": (0.0, 0.0),
            "asset_cfg": SceneEntityCfg("robot")
        }
    )


@configclass
class RewardsCfg:
    # 🌟 绑定我们自己手写的奖励逻辑 (通过 params 把对象传给函数)
    reaching = RewardTermCfg(func=custom_rewards.reaching_distance_reward, weight=1.0, params={"object_cfg": SceneEntityCfg("u_disk")})
    lifting = RewardTermCfg(func=custom_rewards.object_is_lifted, weight=50.0, params={"object_cfg": SceneEntityCfg("u_disk"), "height_threshold": 0.15})
    action_penalty = RewardTermCfg(func=custom_rewards.action_penalty, weight=-0.01)


@configclass
class TerminationsCfg:
    # 简单粗暴：超时直接重置回合
    time_out = TerminationTermCfg(func=mdp.time_out, time_out=True)


@configclass
class GraspEnvCfg(ManagerBasedRLEnvCfg):
    def __post_init__(self):
        # 视角设置：让你一打开就能看到宏大的网格矩阵
        self.viewer.eye = [2.5, 2.5, 2.5]
        self.viewer.lookat = [0.0, 0.0, 0.0]
        
        # 物理配置
        self.decimation = 4
        self.episode_length_s = 10.0
        
        # 挂载所有模块 (注意这里设置了 300 个并行环境)
        self.scene: GraspSceneCfg = GraspSceneCfg(num_envs=300, env_spacing=2.5)
        self.observations: ObservationsCfg = ObservationsCfg()
        self.actions: ActionsCfg = ActionsCfg()
        self.events: EventCfg = EventCfg()
        self.rewards: RewardsCfg = RewardsCfg()
        self.terminations: TerminationsCfg = TerminationsCfg()