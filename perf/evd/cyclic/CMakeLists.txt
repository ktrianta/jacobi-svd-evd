#add_executable(evd_cyclic_perf evd_cyclic_perf.cpp)
#target_link_libraries(evd_cyclic_perf evd-cyclic-lib utils-lib perf-lib ${CMAKE_BINARY_DIR}/pcm/libPCM.a m)
#install(TARGETS evd_cyclic_perf DESTINATION ${EVD_BENCHMARK_BIN_DIR})

message("${Yellow}-- EVD Benchmarks:${ColourReset}")
add_executable(evd_cyclic_vectorize_perf evd_cyclic_vectorize_perf.cpp)
target_link_libraries(evd_cyclic_vectorize_perf evd-cyclic-lib utils-lib perf-lib ${CMAKE_BINARY_DIR}/pcm/libPCM.a m)
install(TARGETS evd_cyclic_vectorize_perf DESTINATION ${EVD_BENCHMARK_BIN_DIR})


add_executable(evd_cyclic_perf_org evd_cyclic_perf_org.cpp)
target_link_libraries(evd_cyclic_perf_org evd-cyclic-lib utils-lib perf-lib ${CMAKE_BINARY_DIR}/pcm/libPCM.a m)
install(TARGETS evd_cyclic_perf_org DESTINATION ${EVD_BENCHMARK_BIN_DIR})

add_executable(evd_cyclic_vectorize_raj_perf evd_cyclic_vectorize_raj_perf.cpp)
target_link_libraries(evd_cyclic_vectorize_raj_perf evd-cyclic-lib utils-lib perf-lib ${CMAKE_BINARY_DIR}/pcm/libPCM.a m)
install(TARGETS evd_cyclic_vectorize_raj_perf DESTINATION ${EVD_BENCHMARK_BIN_DIR})
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

# Benchmark EVD Outer Unroll
message("${Yellow}---- Base${ColourReset}")
add_executable("evd-outer-unroll" outer_unroll_evd.cpp)
target_link_libraries("evd-outer-unroll" evd-cyclic-lib utils-lib perf-lib m)
install(TARGETS "evd-outer-unroll" DESTINATION ${EVD_BENCHMARK_BIN_DIR})

# Benchmark EVD Outer Unroll Vectorized
message("${Yellow}---- Base${ColourReset}")
add_executable("evd-outer-unroll-vectorized" outer_unroll_vectorized_evd.cpp)
target_link_libraries("evd-outer-unroll-vectorized" evd-cyclic-lib utils-lib perf-lib m)
install(TARGETS "evd-outer-unroll-vectorized" DESTINATION ${EVD_BENCHMARK_BIN_DIR})

# Benchmark EVD Oneloop
message("${Yellow}---- Base${ColourReset}")
add_executable("evd-oneloop" oneloop_evd.cpp)
target_link_libraries("evd-oneloop" evd-cyclic-lib utils-lib perf-lib m)
install(TARGETS "evd-oneloop" DESTINATION ${EVD_BENCHMARK_BIN_DIR})

# Benchmark EVD Oneloop Vectorized
message("${Yellow}---- Base${ColourReset}")
add_executable("evd-oneloop-vectorized" oneloop_vectorized_evd.cpp)
target_link_libraries("evd-oneloop-vectorized" evd-cyclic-lib utils-lib perf-lib m)
install(TARGETS "evd-oneloop-vectorized" DESTINATION ${EVD_BENCHMARK_BIN_DIR})

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

# Benchmark EVD Blocked Vectorized
message("${Yellow}---- Base${ColourReset}")
add_executable("evd-blocked-vectorized" blocked_vectorized_evd.cpp)
target_link_libraries("evd-blocked-vectorized" evd-cyclic-lib utils-lib perf-lib m)
install(TARGETS "evd-blocked-vectorized" DESTINATION ${EVD_BENCHMARK_BIN_DIR})

# Benchmark EVD Blocked less copy Vectorized
message("${Yellow}---- Base${ColourReset}")
add_executable("evd-blocked-less-copy-vectorized" blocked_vectorized_evd_less_copy.cpp)
target_link_libraries("evd-blocked-less-copy-vectorized" evd-cyclic-lib utils-lib perf-lib m)
install(TARGETS "evd-blocked-less-copy-vectorized" DESTINATION ${EVD_BENCHMARK_BIN_DIR})
