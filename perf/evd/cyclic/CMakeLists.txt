add_executable(evd_cyclic_perf evd_cyclic_perf.cpp)
target_link_libraries(evd_cyclic_perf evd-cyclic-lib utils-lib perf-lib m)
install(TARGETS evd_cyclic_perf DESTINATION ${EVD_BENCHMARK_BIN_DIR})

message("${Yellow}-- EVD Benchmarks:${ColourReset}")
add_executable(evd_cyclic_vectorize_perf evd_cyclic_vectorize_perf.cpp)
target_link_libraries(evd_cyclic_vectorize_perf evd-cyclic-lib utils-lib perf-lib m)
install(TARGETS evd_cyclic_vectorize_perf DESTINATION ${EVD_BENCHMARK_BIN_DIR})

# Benchmark EVD base
message("${Yellow}---- Base${ColourReset}")
add_executable("evd-base" base_evd.cpp)
target_link_libraries("evd-base" evd-cyclic-lib utils-lib perf-lib m)
install(TARGETS "evd-base" DESTINATION ${EVD_BENCHMARK_BIN_DIR})

# Benchmark EVD Vectorized
message("${Yellow}---- Base${ColourReset}")
add_executable("evd-vectorized" vectorized_evd.cpp)
target_link_libraries("evd-vectorized" evd-cyclic-lib utils-lib perf-lib m)
install(TARGETS "evd-vectorized" DESTINATION ${EVD_BENCHMARK_BIN_DIR})

# Benchmark EVD Blocked
message("${Yellow}---- Base${ColourReset}")
add_executable("evd-blocked" blocked_evd.cpp)
target_link_libraries("evd-blocked" evd-cyclic-lib utils-lib perf-lib m)
install(TARGETS "evd-blocked" DESTINATION ${EVD_BENCHMARK_BIN_DIR})

# Benchmark EVD Blocked less copy
message("${Yellow}---- Base${ColourReset}")
add_executable("evd-blocked-less-copy" blocked_evd_less_copy.cpp)
target_link_libraries("evd-blocked-less-copy" evd-cyclic-lib utils-lib perf-lib m)
install(TARGETS "evd-blocked-less-copy" DESTINATION ${EVD_BENCHMARK_BIN_DIR})
