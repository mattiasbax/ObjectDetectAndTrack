cmake_minimum_required(VERSION 3.19)
project(MyProject)

include(FetchContent)
FetchContent_Declare(
        googletest
        # Specify the commit you depend on and update it regularly.
        URL https://github.com/google/googletest/archive/609281088cfefc76f9d0ce82e1ff6c30cc3591e5.zip
)
# For Windows: Prevent overriding the parent project's compiler/linker settings
set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(googletest)

# Now simply link against gtest or gtest_main as needed. Eg
add_executable(ApplicationTests src/main.cpp
                                src/ApplicationTests.cpp
                                ../../src/MyClass.cpp)

target_link_libraries(ApplicationTests gtest_main)
add_test(NAME example_test COMMAND ApplicationTests)