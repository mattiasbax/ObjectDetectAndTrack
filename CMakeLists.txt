cmake_minimum_required(VERSION 3.19)
project(LearningOpenCV)

set(CMAKE_CXX_STANDARD 17)
set(GIT_FOUND FALSE)
FIND_PACKAGE(OpenCV REQUIRED)

add_subdirectory(src/)
add_subdirectory(tst/)
add_subdirectory(ext/)