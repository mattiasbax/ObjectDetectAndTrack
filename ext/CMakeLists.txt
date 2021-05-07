# DOWNLOAD ALL THE SUBMODULES
find_package(Git QUIET)
if(GIT_FOUND)
    # Update submodules as needed
    message(STATUS "Submodule update")
    execute_process(COMMAND ${GIT_EXECUTABLE} submodule update --init --recursive
            WORKING_DIRECTORY ${PROJECT_SOURCE_DIR}
            RESULT_VARIABLE GIT_SUBMOD_RESULT)
    if(NOT GIT_SUBMOD_RESULT EQUAL "0")
        message(FATAL_ERROR "git submodule update --init failed with ${GIT_SUBMOD_RESULT}")
    endif()
endif()

# ADD ALL EXTERNAL DEPENDANCIES
if(NOT EXISTS "${PROJECT_SOURCE_DIR}/ext/googletest/CMakeLists.txt")
    message(FATAL_ERROR "The submodule googletest was not downloaded. Submodule turned of or failed.")
endif()
add_subdirectory(googletest/)

#if(NOT EXISTS "${PROJECT_SOURCE_DIR}/ext/matplotlibcpp/CMakeLists.txt")
#    message(FATAL_ERROR "The submodule matplotlibcpp was not downloaded. Submodule turned of or failed.")
#endif()
#add_subdirectory(matplotlibcpp/)

