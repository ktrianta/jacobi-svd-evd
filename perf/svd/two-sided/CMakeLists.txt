message("${Yellow}-- SVD Benchmarks:${ColourReset}")

# Benchmark SVD base
message("${Yellow}---- Base${ColourReset}")
add_executable("svd-base" base.cpp)
target_link_libraries("svd-base" two-sided-svd-lib utils-lib perf-lib m)
install(TARGETS "svd-base" DESTINATION ${SVD_BENCHMARK_BIN_DIR})

# Benchmark SVD blocked
message("${Yellow}---- Blocked${ColourReset}")
add_executable("svd-blocked" blocked.cpp)
target_link_libraries("svd-blocked" two-sided-svd-lib utils-lib perf-lib m)
install(TARGETS "svd-blocked" DESTINATION ${SVD_BENCHMARK_BIN_DIR})
