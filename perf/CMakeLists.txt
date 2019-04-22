cmake_minimum_required(VERSION 3.9.2)

# Performance benchmarks install directory
set(BENCHMARK_BIN_DIR "${PROJECT_SOURCE_DIR}/bin/benchmark")

add_subdirectory(libs)
add_subdirectory(evd)
add_subdirectory(svd)
