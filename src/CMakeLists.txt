cmake_minimum_required(VERSION 3.19)

SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -pthread")
set(EXECUTABLE_NAME ${PROJECT_NAME}_app)
add_executable( ${EXECUTABLE_NAME}  main.cpp
        ObjectTrackerHandler.cpp
                                    ObjectTrackerFactory.cpp ObjectDetectorHandler.cpp ObjectDetectorHandler.h ObjectDetectorHandler.cpp)

target_link_libraries(${EXECUTABLE_NAME} opencv_core opencv_highgui opencv_imgcodecs opencv_tracking opencv_dnn)