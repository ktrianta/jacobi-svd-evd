cmake_minimum_required(VERSION 3.12)

# Includes
include(CTest)

# Download and unpack googletest at configure time.
configure_file(CMakeLists.txt.in googletest-download/CMakeLists.txt)
execute_process(COMMAND ${CMAKE_COMMAND} -G "${CMAKE_GENERATOR}" .
  RESULT_VARIABLE result
  WORKING_DIRECTORY ${CMAKE_BINARY_DIR}/googletest-download )
if(result)
  message(FATAL_ERROR "CMake step for googletest failed: ${result}")
endif()
execute_process(COMMAND ${CMAKE_COMMAND} --build .
  RESULT_VARIABLE result
  WORKING_DIRECTORY ${CMAKE_BINARY_DIR}/googletest-download )
if(result)
  message(FATAL_ERROR "Build step for googletest failed: ${result}")
endif()

# Prevent overriding the parent project's compiler/linker settings on Windows.
set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)

# Define the gtest and gtest_main targets.
add_subdirectory(${CMAKE_BINARY_DIR}/googletest-src
                 ${CMAKE_BINARY_DIR}/googletest-build
                 EXCLUDE_FROM_ALL)

# Set C++ standard and compiler flags.
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Wextra -Werror -pthread -std=c++11 -save-temps -Wno-unused-parameter -Wno-unused-variable -Wno-deprecated-copy")
set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS} -DNDEBUG -O3 -mavx2 -mfma -fno-tree-vectorize")

# If cmake is not passed the CMAKE_BUILD_TYPE=Debug option, default to Release.
# Note that if the CMAKE_BUILD_TYPE option is passed when running cmake, its value is cached
# for subsequent runs. To clear it delete the cmake cache or set it to another build type.
if (NOT CMAKE_BUILD_TYPE STREQUAL "Debug")
    set(CMAKE_BUILD_TYPE "Release")
endif()

if(NOT WIN32)
  string(ASCII 27 Esc)
  set(ColourReset "${Esc}[m")
  set(Yellow      "${Esc}[33m")
endif()

# Print the build type in yellow color to make it visible.
message("${Yellow}Build type set to ${CMAKE_BUILD_TYPE}${ColourReset}")

# Project subdirectories
add_subdirectory(src)
add_subdirectory(perf)
enable_testing()
add_subdirectory(test)
