cmake_minimum_required(VERSION 3.19)
project(LearningOpenCV)

set(CMAKE_CXX_STANDARD 17)
set(OpenCV_DIR C:/Users/Mattias/CLionProjects/opencv/cmake-build-release/install)
FIND_PACKAGE(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})

add_subdirectory(src/)
add_subdirectory(tst/)
add_subdirectory(ext/)