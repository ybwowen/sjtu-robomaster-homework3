cmake_minimum_required( VERSION 2.8 )
project(armor)

set(CMAKE_CXX_FLAGS "-std=c++14")

include_directories("/usr/include/eigen3")
find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})
include_directories("usr/include/yaml-cpp")

add_executable(armor main.cpp)
target_link_libraries(armor ${OpenCV_LIBS})