add_executable(evd_cyclic_perf evd_cyclic_perf.cpp)
target_link_libraries(evd_cyclic_perf evd-cyclic-lib utils-lib perf-lib m)
install(TARGETS evd_cyclic_perf DESTINATION ${EVD_BENCHMARK_BIN_DIR})

add_executable(evd_cyclic_vectorize_perf evd_cyclic_vectorize_perf.cpp)
target_link_libraries(evd_cyclic_vectorize_perf evd-cyclic-lib utils-lib perf-lib m)
install(TARGETS evd_cyclic_vectorize_perf DESTINATION ${EVD_BENCHMARK_BIN_DIR})
