cmake_minimum_required(VERSION 3.9.2)

# Performance benchmarks install directory
set(SVD_BENCHMARK_BIN_DIR "${PROJECT_SOURCE_DIR}/bin/benchmark/svd")
set(EVD_BENCHMARK_BIN_DIR "${PROJECT_SOURCE_DIR}/bin/benchmark/evd")

# Eigen library directory
set(EIGEN_LIB_DIR "${PROJECT_SOURCE_DIR}/perf/eigen")

add_subdirectory(libs)
add_subdirectory(evd)
add_subdirectory(svd)
