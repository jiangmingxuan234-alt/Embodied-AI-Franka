# IK Grasp Data Collection (Archived)

早期尝试使用传统运动学逆解（Inverse Kinematics）进行自动化抓取数据采集的脚本。

* **实验现象**：在 Isaac 仿真环境中，纯依靠 IK 规划的抓取动作在面对动态扰动时鲁棒性极差，且难以收集到高质量、带有人类纠偏意图的顺滑轨迹数据。
* **结论与转向**：传统解析解方法在复杂接触任务中存在瓶颈。本项目最终全面转向 Teleoperation（遥操作）收集专家数据，并采用基于学习的视觉运动策略（Visuomotor Policy）来实现更柔顺的端到端控制。
