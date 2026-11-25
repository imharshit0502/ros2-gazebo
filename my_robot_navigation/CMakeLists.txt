cmake_minimum_required(VERSION 3.8)
project(my_robot_navigation)

find_package(ament_cmake REQUIRED)

install(
  DIRECTORY launch config rviz maps
  DESTINATION share/${PROJECT_NAME}/
)

# Python nodes -> `ros2 run my_robot_navigation mapping_tour.py`
install(
  PROGRAMS scripts/mapping_tour.py scripts/shelf_patrol.py
  DESTINATION lib/${PROJECT_NAME}
)

ament_package()
