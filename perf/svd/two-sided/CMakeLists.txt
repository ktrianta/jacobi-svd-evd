add_executable(svd_two_sided_perf svd_perf.cpp)
target_link_libraries(svd_two_sided_perf two-sided-svd-lib utils-lib perf-lib m)
install(TARGETS svd_two_sided_perf DESTINATION ${SVD_BENCHMARK_BIN_DIR})
