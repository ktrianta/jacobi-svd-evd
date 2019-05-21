add_executable(svd_one_sided_perf svd_perf.cpp)
target_link_libraries(svd_one_sided_perf svd-lib utils-lib perf-lib ${CMAKE_BINARY_DIR}/pcm/libPCM.a m)
install(TARGETS svd_one_sided_perf DESTINATION ${SVD_BENCHMARK_BIN_DIR})
