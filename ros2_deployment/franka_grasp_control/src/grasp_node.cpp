#include <chrono>
#include <memory>
#include <cmath> // 引入数学库计算余弦
#include "rclcpp/rclcpp.hpp"
#include "geometry_msgs/msg/pose.hpp"

using namespace std::chrono_literals;

class FrankaXYZNode : public rclcpp::Node
{
public:
  FrankaXYZNode() : Node("franka_xyz_node"), step_count_(0)
  {
    publisher_ = this->create_publisher<geometry_msgs::msg::Pose>("/franka/target_pose", 10);
    timer_ = this->create_wall_timer(50ms, std::bind(&FrankaXYZNode::timer_callback, this));
    RCLCPP_INFO(this->get_logger(), "🧠 S-Curve 平滑控制大脑启动！准备执行高级温柔抓取...");
  }

private:
  // 🌟 核心算法：S-Curve 平滑插值函数 (彻底告别线性碰撞)
  double smooth_step(double start, double end, double progress) {
      if (progress <= 0.0) return start;
      if (progress >= 1.0) return end;
      // 利用余弦曲线生成 0 到 1 的平滑过渡 (两头慢，中间快)
      double cosine_progress = 0.5 * (1.0 - std::cos(M_PI * progress));
      return start + (end - start) * cosine_progress;
  }

  void timer_callback()
  {
    auto message = geometry_msgs::msg::Pose();
    
    // 锁定 XY 坐标在 U 盘正上方 (如需微调，改这里)
    message.position.x = 0.26;
    message.position.y = -0.003;
    
    double time_sec = step_count_ * 0.05;

    // 🎯 降速 + 平滑状态机
    if (time_sec < 3.0) {
        // 【阶段 1：起步高空对齐】(0~3秒) - 从初始位置平滑对齐到正上方 0.50
        message.position.z = 0.50; 
        message.orientation.w = 1.0; // 张开夹爪
    } 
    else if (time_sec < 8.0) {
        // 【阶段 2：温柔下探】(3~8秒) - 花整整 5 秒钟，用 S 曲线像羽毛一样降落
        double progress = (time_sec - 3.0) / 5.0; 
        // 0.145 为抓取高度，若太高抓空可改小(如0.135)，若撞击可改大(如0.155)
        message.position.z = smooth_step(0.50, 0.145, progress); 
        message.orientation.w = 1.0; 
    } 
    else if (time_sec < 9.5) {
        // 【阶段 3：停顿捏合】(8~9.5秒) - 给足 1.5 秒让物理引擎计算夹爪受力
        message.position.z = 0.145;
        message.orientation.w = -1.0; // 闭合夹爪！
    } 
    else if (time_sec < 14.0) {
        // 【阶段 4：平滑拔起】(9.5~14秒) - 捏紧 U 盘，用 S 曲线温柔抬升到 0.45
        double progress = (time_sec - 9.5) / 4.5;
        message.position.z = smooth_step(0.145, 0.45, progress); 
        message.orientation.w = -1.0; // 死死捏住！
    } 
    else {
        // 【阶段 5：任务完成】(14秒之后) - 悬停展示
        message.position.z = 0.45;
        message.orientation.w = -1.0;
    }

    publisher_->publish(message);
    step_count_++;
  }

  rclcpp::TimerBase::SharedPtr timer_;
  rclcpp::Publisher<geometry_msgs::msg::Pose>::SharedPtr publisher_;
  int step_count_;
};

int main(int argc, char * argv[])
{
  rclcpp::init(argc, argv);
  rclcpp::spin(std::make_shared<FrankaXYZNode>());
  rclcpp::shutdown();
  return 0;
}