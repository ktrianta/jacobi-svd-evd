add_executable(eigen_svd_perf svd_perf.cpp)
target_link_libraries(eigen_svd_perf two-sided-svd-lib utils-lib perf-lib m)
target_include_directories(eigen_svd_perf PRIVATE ${EIGEN_LIB_DIR})
install(TARGETS eigen_svd_perf DESTINATION ${SVD_BENCHMARK_BIN_DIR})
