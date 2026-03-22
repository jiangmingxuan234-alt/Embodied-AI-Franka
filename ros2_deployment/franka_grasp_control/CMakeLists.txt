cmake_minimum_required(VERSION 3.8)
project(franka_grasp_control)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(geometry_msgs REQUIRED)

add_executable(grasp_node src/grasp_node.cpp)
ament_target_dependencies(grasp_node rclcpp geometry_msgs)

install(TARGETS
  grasp_node
  DESTINATION lib/${PROJECT_NAME}
)

install(PROGRAMS
  scripts/franka_joint_publisher.py
  scripts/franka_hybrid_policy_node.py
  scripts/franka_customplay_ros_node.py
  DESTINATION lib/${PROJECT_NAME}
)

install(DIRECTORY config
  DESTINATION share/${PROJECT_NAME}
)

if(BUILD_TESTING)
  find_package(ament_lint_auto REQUIRED)
  set(ament_cmake_copyright_FOUND TRUE)
  set(ament_cmake_cpplint_FOUND TRUE)
  ament_lint_auto_find_test_dependencies()
endif()

ament_package()
