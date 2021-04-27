cmake_minimum_required(VERSION 3.19)

include(FetchContent)
FetchContent_Declare(
        googletest
        # Specify the commit you depend on and update it regularly.
        URL https://github.com/google/googletest/archive/release-1.10.0.zip
)
# For Windows: Prevent overriding the parent project's compiler/linker settings
set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(googletest)

add_executable(${PROJECT_NAME}_app_tests src/main.cpp
                                         src/ApplicationTests.cpp
                                         ../../src/MyClass.cpp)

target_link_libraries(${PROJECT_NAME}_app_tests gtest_main)
add_test(NAME example_test COMMAND ApplicationTests)