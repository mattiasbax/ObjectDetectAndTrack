cmake_minimum_required(VERSION 3.19)

set(EXECUTABLE_NAME ${PROJECT_NAME}_app)
add_executable( ${EXECUTABLE_NAME} main.cpp
                                   MyClass.cpp)