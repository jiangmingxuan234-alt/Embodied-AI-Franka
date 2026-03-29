import math
import torch
import isaaclab.envs.mdp as mdp
from isaaclab.envs import ManagerBasedRLEnv, ManagerBasedRLEnvCfg
from isaaclab.managers import ObservationGroupCfg, ObservationTermCfg, ActionTermCfg, RewardTermCfg, SceneEntityCfg, EventTermCfg, TerminationTermCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.utils import configclass
from isaaclab.assets import ArticulationCfg, RigidObjectCfg, AssetBaseCfg

# 适配最新版本的 spawner 和 schemas
import isaaclab.sim as sim_utils

# 资产加载（确保已安装 isaaclab_assets）
from isaaclab_assets.robots.franka import FRANKA_PANDA_CFG
from .mdp import rewards as custom_rewards

GOAL_POSITION = [0.40, 0.0, 0.40]


def obs_37dim(env: ManagerBasedRLEnv, object_cfg: SceneEntityCfg, robot_cfg: SceneEntityCfg) -> torch.Tensor:
    """37 维观测，与 ACT 训练时完全一致：
    joint_pos(9) + joint_vel(9) + rel(3) + obj_pos(3) + ee_pos(3) + ee_quat(4) + goal_rel(3) + phase(3)
    """
    robot = env.scene[robot_cfg.name]
    obj = env.scene[object_cfg.name]

    j_pos = robot.data.joint_pos[:, :9]   # (N, 9)
    j_vel = robot.data.joint_vel[:, :9]   # (N, 9)

    hand_idx = robot.find_bodies("panda_hand")[0][0]      # scalar index
    ee_pos = robot.data.body_state_w[:, hand_idx, :3]    # (N, 3)
    ee_quat = robot.data.body_state_w[:, hand_idx, 3:7]  # (N, 4) wxyz

    obj_pos = obj.data.root_pos_w[:, :3]   # (N, 3)
    rel = obj_pos - ee_pos                 # (N, 3)

    goal = torch.tensor(GOAL_POSITION, device=env.device).unsqueeze(0).expand(env.num_envs, -1)
    goal_rel = goal - obj_pos             # (N, 3)

    # phase encoding
    on_table = (obj_pos[:, 2] <= 0.09).float()
    near_goal = (torch.norm(obj_pos - goal, dim=-1) <= 0.06).float()
    approaching = (1.0 - on_table) * (1.0 - near_goal)
    phase = torch.stack([on_table, approaching, near_goal * (1.0 - on_table)], dim=-1)  # (N, 3)

    return torch.cat([j_pos, j_vel, rel, obj_pos, ee_pos, ee_quat, goal_rel, phase], dim=-1)  # (N, 37)

@configclass
class GraspSceneCfg(InteractiveSceneCfg):
    # 1. 地面与灯光 (使用 AssetBaseCfg 包装)
    ground = AssetBaseCfg(prim_path="/World/ground", spawn=sim_utils.GroundPlaneCfg())
    light = AssetBaseCfg(prim_path="/World/light", spawn=sim_utils.DomeLightCfg(intensity=2000.0))

    # 2. 机器人：Franka Panda
    robot: ArticulationCfg = FRANKA_PANDA_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")

    # 3. U 盘：红色高摩擦力长方体
    u_disk = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/UDisk",
        spawn=sim_utils.CuboidCfg(
            size=[0.12, 0.03, 0.025],
            rigid_props=sim_utils.RigidBodyPropertiesCfg(disable_gravity=False),
            mass_props=sim_utils.MassPropertiesCfg(mass=0.02), # 🌟 修正：质量属性位置
            collision_props=sim_utils.CollisionPropertiesCfg(),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(1.0, 0.0, 0.0)),
            physics_material=sim_utils.RigidBodyMaterialCfg(static_friction=2.0, dynamic_friction=2.0)
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0.3, 0.0, 0.02)),
    )

@configclass
class ActionsCfg:
    arm_action = mdp.JointPositionActionCfg(
        asset_name="robot", joint_names=["panda_joint.*", "panda_finger.*"], scale=1.0, use_default_offset=True
    )


@configclass
class ActionsFinetuneCfg:
    """Fine-tune 动作配置：降低 scale 限制动作幅度，减少抖动"""
    arm_action = mdp.JointPositionActionCfg(
        asset_name="robot", joint_names=["panda_joint.*", "panda_finger.*"], scale=1.0, use_default_offset=True
    )

@configclass
class ObservationsCfg:
    @configclass
    class PolicyCfg(ObservationGroupCfg):
        joint_pos = ObservationTermCfg(func=mdp.joint_pos_rel, params={"asset_cfg": SceneEntityCfg("robot")})
        joint_vel = ObservationTermCfg(func=mdp.joint_vel_rel, params={"asset_cfg": SceneEntityCfg("robot")})
        object_pos = ObservationTermCfg(func=mdp.root_pos_w, params={"asset_cfg": SceneEntityCfg("u_disk")})
        object_quat = ObservationTermCfg(func=mdp.root_quat_w, params={"asset_cfg": SceneEntityCfg("u_disk")})

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = True

    policy: PolicyCfg = PolicyCfg()


@configclass
class ObservationsFinetuneCfg:
    """37 维观测，与 ACT 训练时完全一致。"""
    @configclass
    class PolicyCfg(ObservationGroupCfg):
        obs_full = ObservationTermCfg(
            func=obs_37dim,
            params={"object_cfg": SceneEntityCfg("u_disk"), "robot_cfg": SceneEntityCfg("robot")},
        )

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = True

    policy: PolicyCfg = PolicyCfg()

@configclass
class EventCfg:
    randomize_object = EventTermCfg(
        func=mdp.reset_root_state_uniform,
        mode="reset",
        params={
            # 🌟 收缩范围：确保机械臂绝对够得着
            "pose_range": {"x": (0.25, 0.35), "y": (-0.1, 0.1), "yaw": (-math.pi, math.pi)},
            "velocity_range": {
                "x": (0.0, 0.0), "y": (0.0, 0.0), "z": (0.0, 0.0),
                "roll": (0.0, 0.0), "pitch": (0.0, 0.0), "yaw": (0.0, 0.0)
            },
            "asset_cfg": SceneEntityCfg("u_disk")
        }
    )
    reset_robot = EventTermCfg(
        func=mdp.reset_joints_by_scale, 
        mode="reset", 
        params={
            "position_range": (1.0, 1.0), 
            "velocity_range": (0.0, 0.0), 
            "asset_cfg": SceneEntityCfg("robot")
        }
    )

@configclass
class RewardsCfg:
    """奖励配置（纯 RL 从零训练版本）"""
    # 基础奖励
    reaching = RewardTermCfg(func=custom_rewards.reaching_reward, weight=1.0, params={"object_cfg": SceneEntityCfg("u_disk")})
    grasping = RewardTermCfg(func=custom_rewards.grasping_reward, weight=10.0, params={"object_cfg": SceneEntityCfg("u_disk")})
    lift_to_target = RewardTermCfg(func=custom_rewards.lift_to_target_reward, weight=20.0, params={"object_cfg": SceneEntityCfg("u_disk"), "target_height": 0.25})
    
    # 🌟 强行注入"人类抓取逻辑" (权重给足，强迫它对齐)
    top_down = RewardTermCfg(func=custom_rewards.top_down_posture_reward, weight=5.0)
    hover_above = RewardTermCfg(func=custom_rewards.hover_above_reward, weight=5.0, params={"object_cfg": SceneEntityCfg("u_disk")})
    
    # 惩罚项
    action_rate_penalty = RewardTermCfg(func=custom_rewards.action_rate_penalty, weight=-0.05)
    drop_penalty = RewardTermCfg(func=custom_rewards.drop_penalty, weight=-10.0, params={"object_cfg": SceneEntityCfg("u_disk")})


@configclass
class RewardsFinetuneCfg:
    """Fine-tune 专用奖励：高斯核 + 三斧惩罚"""
    # 基础奖励（高斯核）
    reaching = RewardTermCfg(func=custom_rewards.reaching_reward, weight=1.5, params={"object_cfg": SceneEntityCfg("u_disk")})
    grasping = RewardTermCfg(func=custom_rewards.grasping_reward, weight=5.0, params={"object_cfg": SceneEntityCfg("u_disk")})
    lift_to_target = RewardTermCfg(func=custom_rewards.lift_to_target_reward, weight=25.0, params={"object_cfg": SceneEntityCfg("u_disk"), "target_height": 0.25})

    # 姿态奖励
    top_down = RewardTermCfg(func=custom_rewards.top_down_posture_reward, weight=3.0)
    hover_above = RewardTermCfg(func=custom_rewards.hover_above_reward, weight=3.0, params={"object_cfg": SceneEntityCfg("u_disk")})

    # 夹爪控制
    gripper_close = RewardTermCfg(func=custom_rewards.gripper_close_near_object, weight=5.0, params={"object_cfg": SceneEntityCfg("u_disk")})

    # 三斧惩罚
    action_rate_penalty = RewardTermCfg(func=custom_rewards.action_rate_penalty, weight=-0.5)
    joint_accel_penalty = RewardTermCfg(func=custom_rewards.joint_acceleration_penalty, weight=-0.01)
    action_l2_penalty = RewardTermCfg(func=custom_rewards.action_l2_penalty, weight=-0.005)
    drop_penalty = RewardTermCfg(func=custom_rewards.drop_penalty, weight=-10.0, params={"object_cfg": SceneEntityCfg("u_disk")})
@configclass
class TerminationsCfg:
    time_out = TerminationTermCfg(func=mdp.time_out, time_out=True)

@configclass
class GraspEnvCfg(ManagerBasedRLEnvCfg):
    """环境配置（纯 RL 从零训练版本）"""
    def __post_init__(self):
        self.viewer.eye = [2.0, 2.0, 2.0]
        self.decimation = 4
        # 🌟 不赶时间，给足 20 秒去瞄准
        self.episode_length_s = 20.0
        self.scene: GraspSceneCfg = GraspSceneCfg(num_envs=300, env_spacing=2.5)
        self.observations: ObservationsCfg = ObservationsCfg()
        self.actions: ActionsCfg = ActionsCfg()
        self.events: EventCfg = EventCfg()
        self.rewards: RewardsCfg = RewardsCfg()
        self.terminations: TerminationsCfg = TerminationsCfg()


@configclass
class GraspEnvFinetuneCfg(ManagerBasedRLEnvCfg):
    """
    Fine-tune 专用环境配置
    
    与纯 RL 版本的区别：
    1. 使用 RewardsFinetuneCfg（连续密集奖励）
    2. 增加物体随机化范围（提高泛化性）
    3. 增加环境数量（更稳定的梯度估计）
    """
    def __post_init__(self):
        self.viewer.eye = [2.0, 2.0, 2.0]
        self.decimation = 4
        self.episode_length_s = 20.0
        # 增加环境数量以获得更稳定的梯度
        self.scene: GraspSceneCfg = GraspSceneCfg(num_envs=400, env_spacing=2.5)
        self.observations: ObservationsFinetuneCfg = ObservationsFinetuneCfg()
        self.actions: ActionsFinetuneCfg = ActionsFinetuneCfg()
        # Fine-tune 使用更大的物体随机化范围
        self.events: EventCfg = EventFinetuneCfg()
        self.rewards: RewardsFinetuneCfg = RewardsFinetuneCfg()
        self.terminations: TerminationsCfg = TerminationsCfg()


@configclass
class EventFinetuneCfg(EventCfg):
    """
    Fine-tune 专用事件配置
    
    增加物体随机化范围，提高策略泛化性
    """
    randomize_object = EventTermCfg(
        func=mdp.reset_root_state_uniform,
        mode="reset",
        params={
            # 🌟 扩大范围：测试泛化能力
            "pose_range": {"x": (0.20, 0.45), "y": (-0.20, 0.20), "yaw": (-math.pi, math.pi)},
            "velocity_range": {
                "x": (0.0, 0.0), "y": (0.0, 0.0), "z": (0.0, 0.0),
                "roll": (0.0, 0.0), "pitch": (0.0, 0.0), "yaw": (0.0, 0.0)
            },
            "asset_cfg": SceneEntityCfg("u_disk")
        }
    )
    reset_robot = EventTermCfg(
        func=mdp.reset_joints_by_scale,
        mode="reset",
        params={
            "position_range": (1.0, 1.0),
            "velocity_range": (0.0, 0.0),
            "asset_cfg": SceneEntityCfg("robot")
        }
    )