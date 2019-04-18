cmake_minimum_required(VERSION 3.9.2)

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Wextra -Werror -std=c++11")
set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS} -g -O0")
set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS} -O3 -march=avx2 -mfma -fno-tree-vectorize")

set(BENCHMARK_BIN_DIR "${PROJECT_SOURCE_DIR}/bin/benchmark")

add_subdirectory(libs)
add_subdirectory(evd)
add_subdirectory(svd)
