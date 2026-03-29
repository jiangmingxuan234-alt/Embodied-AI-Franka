# ROS2 部署：Franka 抓取控制

将训练好的策略（BC / ACT）部署到 ROS2，通过 Isaac Sim 桥接节点实现仿真验证或实机控制。

## 架构

```
┌─────────────────────────────────────────────────────────────┐
│                    Isaac Sim 仿真环境                         │
│  ┌──────────┐  ┌──────────┐  ┌────────────────────────────┐ │
│  │ Franka   │  │  U-Disk  │  │  PickPlaceController       │ │
│  │ Panda    │  │ (Cuboid) │  │  (Assist 专家控制器)        │ │
│  └──────────┘  └──────────┘  └────────────────────────────┘ │
│                       │                                      │
│              ros2_env_bridge.py                               │
│         发布状态 / 接收指令 / Assist 接管                      │
└───────────────────────┬─────────────────────────────────────┘
                        │ ROS2 Topics
                        ▼
┌─────────────────────────────────────────────────────────────┐
│                    策略节点（三选一）                          │
│  ┌──────────────────────────────────────────────────────┐   │
│  │ franka_hybrid_policy_node.py   (BC + KNN 辅助)       │   │
│  │ franka_customplay_ros_node.py  (BC + 动作整形)       │   │
│  │ franka_act_policy_node.py      (ACT + Temporal Ens.) │   │
│  └──────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────┘
```

## ROS2 话题接口

| 话题 | 类型 | 方向 | 说明 |
|------|------|------|------|
| `/joint_states` | sensor_msgs/JointState | 桥接 → 策略 | 9 维关节位置 + 速度 |
| `/franka/ee_pose` | geometry_msgs/PoseStamped | 桥接 → 策略 | 末端执行器位姿 |
| `/udisk/pose` | geometry_msgs/PoseStamped | 桥接 → 策略 | U 盘位姿 |
| `/joint_command` | sensor_msgs/JointState | 策略 → 桥接 | 9 维目标关节位置 |
| `/assist_start` | std_msgs/Empty | 策略 → 桥接 | 触发专家控制器接管 |
| `/assist_stop` | std_msgs/Empty | 策略 → 桥接 | 归还控制权给策略 |

## 文件结构

```
ros2_deployment/franka_grasp_control/
├── scripts/
│   ├── franka_hybrid_policy_node.py    # BC 混合策略节点
│   ├── franka_customplay_ros_node.py   # BC 纯策略节点
│   ├── franka_act_policy_node.py       # ACT 策略节点
│   └── franka_joint_publisher.py       # 测试用关节发布器
├── config/
│   ├── hybrid_policy.yaml              # BC 混合策略参数
│   ├── customplay_policy.yaml          # BC 纯策略参数
│   └── act_policy.yaml                 # ACT 策略参数
├── src/
│   └── grasp_node.cpp                  # C++ 轨迹规划节点
├── CMakeLists.txt
└── package.xml
```

## 快速启动

### 方案一：BC 混合策略

```bash
# 终端 1：启动 Isaac Sim 桥接
python source/standalone/ros2_env_bridge.py

# 终端 2：启动 BC 策略节点
python ros2_deployment/franka_grasp_control/scripts/franka_hybrid_policy_node.py \
    --ros-args --params-file \
    ros2_deployment/franka_grasp_control/config/hybrid_policy.yaml
```

### 方案二：ACT 策略

```bash
# 终端 1：启动 Isaac Sim 桥接
python source/standalone/ros2_env_bridge.py

# 终端 2：启动 ACT 策略节点
python ros2_deployment/franka_grasp_control/scripts/franka_act_policy_node.py \
    --ros-args --params-file \
    ros2_deployment/franka_grasp_control/config/act_policy.yaml
```

## 策略节点详解

### 1. BC 混合策略 (`franka_hybrid_policy_node.py`)

基于 robomimic BC 模型 + 复杂辅助状态机。

| 特性 | 说明 |
|------|------|
| 模型 | robomimic BC (MLP [256,256,256]) |
| 动作类型 | delta（增量叠加到当前关节） |
| 控制频率 | 20 Hz |
| 辅助机制 | KNN 专家引导 + 卡住检测 + 下压救援 + burst 脉冲 |
| EMA 平滑 | alpha=0.7 |
| 关节限幅 | 0.004 rad/step |
| 规则夹爪 | xy < 10cm 且 z < 14cm 时闭合 |
| 参数数量 | ~60 个可调参数 |

辅助状态机流程：
1. 正常模式 → BC 推理 + EMA 平滑
2. 卡住检测（距离变化 < 0.0006 持续 12 步）→ 切换辅助模式
3. 辅助模式 → 混合 BC(5%) + KNN 专家(95%)
4. 下压区域 → 混合 BC(70%) + 专家(30%)
5. 目标达成或超时 → 退出辅助

### 2. ACT 策略 (`franka_act_policy_node.py`)

基于 ACT 模型，自包含（不依赖 robomimic）。

| 特性 | 说明 |
|------|------|
| 模型 | ACT (CVAE + Transformer, ~3.5M 参数) |
| 动作类型 | 绝对关节位置（直接应用） |
| 控制频率 | 20 Hz |
| Temporal Ensemble | chunk(50步) 加权平均，`w = exp(-0.1 * idx)` |
| 观测归一化 | 输入前归一化到 [-1, 1] |
| 动作反归一化 | `action = 0.5 * (norm + 1.0) * range + min` |
| 夹爪控制 | 模型预测 + 规则保护（xy < 5cm 且 z < 6cm 才闭合） |
| 辅助机制 | 简化卡住检测 → 发 `/assist_start` 事件 |
| 参数数量 | ~15 个 |

### 3. BC 纯策略 (`franka_customplay_ros_node.py`)

轻量版 BC 部署，无 KNN 辅助，使用动作整形替代。

| 特性 | 说明 |
|------|------|
| 控制频率 | 15 Hz |
| 动作整形 | 增益缩放(3.2x) + 精细区检测 + 摇头阻尼 + 死区 + 翻转抑制 |
| 辅助 | 发布 `/assist_start` 事件（外部处理） |

## 桥接节点 (`ros2_env_bridge.py`)

位于 `source/standalone/ros2_env_bridge.py`，运行在 Isaac Sim 内部。

功能：
- 创建仿真世界（Franka + U 盘 + 地面）
- 发布机器人状态（关节、末端、物体位姿）
- 接收 `/joint_command` 控制机械臂
- 接收 `/assist_start` 时切换到内置 PickPlaceController 专家控制
- 接收 `/assist_stop` 时归还控制权

## 配置文件

### `hybrid_policy.yaml` 关键参数

```yaml
control_rate_hz: 20.0
goal_position: [0.40, 0.0, 0.40]
checkpoint_path: logs/robomimic/.../model_epoch_100.pth
dataset_path: logs/robomimic/rmpflow_expert.hdf5
enable_assist: true
assist_stuck_steps: 12
arm_delta_limit: [0.004, 0.004, 0.004, 0.004, 0.004, 0.004, 0.004]
```

### `act_policy.yaml` 关键参数

```yaml
control_rate_hz: 20.0
goal_position: [0.40, 0.0, 0.40]
checkpoint_path: logs/act_policy/act_epoch_300.pt
norm_stats_path: logs/act_policy/norm_stats.npz
temporal_decay: 0.1
target_slew_limit: [0.006, 0.006, 0.006, 0.006, 0.006, 0.006, 0.006]
grasp_xy: 0.05
grasp_z: 0.06
```

## BC vs ACT 部署对比

| | BC 混合策略 | ACT 策略 |
|---|---|---|
| 动作类型 | delta（当前关节 + 增量） | 绝对关节位置 |
| 推理方式 | 单步输出 | chunk(50步) + temporal ensemble |
| 观测处理 | 原始值直接输入 | 归一化到 [-1, 1] |
| 模型依赖 | robomimic + h5py | 纯 PyTorch（自包含） |
| 辅助机制 | KNN + 状态机 + burst 脉冲 | 简化卡住检测 → assist 事件 |
| 参数复杂度 | ~60 个 | ~15 个 |
| 适用场景 | 需要精细工程调优 | 追求简洁部署 |

## 编译安装（可选）

如果需要通过 `ros2 run` 启动：

```bash
cd ros2_deployment
colcon build --packages-select franka_grasp_control
source install/setup.bash
ros2 run franka_grasp_control franka_act_policy_node.py \
    --ros-args --params-file \
    $(ros2 pkg prefix franka_grasp_control)/share/franka_grasp_control/config/act_policy.yaml
```

直接用 python 运行也可以，不需要编译。
