cmake_minimum_required(VERSION 3.19)

set(EXECUTABLE_NAME ${PROJECT_NAME}_app_tests)
add_executable(${EXECUTABLE_NAME} src/main.cpp
                                  src/ApplicationTests.cpp
        )

target_include_directories(${EXECUTABLE_NAME}
                            PUBLIC  ${PROJECT_SOURCE_DIR}/src
                            PUBLIC  ${PROJECT_SOURCE_DIR}/ext/googletest/googletest/include
                            PUBLIC  ${PROJECT_SOURCE_DIR}/ext/googletest/googlemock/include
                            PUBLIC  ${PROJECT_SOURCE_DIR}/ext/matplotlibcpp)

target_link_directories(${EXECUTABLE_NAME}
                        PRIVATE  ${PROJECT_SOURCE_DIR}/ext/googletest/googletest/src
                        PRIVATE  ${PROJECT_SOURCE_DIR}/ext/googletest/googletest/src)

target_link_libraries(${EXECUTABLE_NAME} gtest_main matplotlib_cpp)
#add_test(NAME example_test COMMAND ${PROJECT_NAME}_app_tests)