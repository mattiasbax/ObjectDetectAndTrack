cmake_minimum_required(VERSION 3.19)

set(EXECUTABLE_NAME ${PROJECT_NAME}_app)
add_executable( ${EXECUTABLE_NAME}  main.cpp
                                    MyClass.cpp
                                    ObjectTrackerHandler.cpp
                                    ObjectTrackerFactory.cpp)

target_link_libraries(${EXECUTABLE_NAME} opencv_core opencv_highgui opencv_imgcodecs opencv_video)