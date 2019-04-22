add_executable(evd_classic_perf evd_classic_perf.cpp)
target_link_libraries(evd_classic_perf evd-classic-lib utils-lib perf-lib m)
install(TARGETS evd_classic_perf DESTINATION ${EVD_BENCHMARK_BIN_DIR})
