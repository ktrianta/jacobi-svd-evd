message("${Yellow}-- SVD Benchmarks:${ColourReset}")

# Benchmark SVD base
message("${Yellow}---- Base${ColourReset}")
add_executable("svd-base" base.cpp)
target_link_libraries("svd-base" two-sided-svd-lib utils-lib perf-lib ${CMAKE_BINARY_DIR}/pcm/libPCM.a m)
install(TARGETS "svd-base" DESTINATION ${SVD_BENCHMARK_BIN_DIR})

# Benchmark SVD blocked
message("${Yellow}---- Blocked${ColourReset}")
add_executable("svd-blocked" blocked.cpp)
target_link_libraries("svd-blocked" two-sided-svd-lib utils-lib perf-lib ${CMAKE_BINARY_DIR}/pcm/libPCM.a m)
install(TARGETS "svd-blocked" DESTINATION ${SVD_BENCHMARK_BIN_DIR})

# Benchmark SVD blocked less copy
message("${Yellow}---- Blocked Less Copying${ColourReset}")
add_executable("svd-blocked-less-copy" blocked_less_copy.cpp)
target_link_libraries("svd-blocked-less-copy" two-sided-svd-lib utils-lib perf-lib ${CMAKE_BINARY_DIR}/pcm/libPCM.a m)
install(TARGETS "svd-blocked-less-copy" DESTINATION ${SVD_BENCHMARK_BIN_DIR})

# Benchmark SVD blocked less copy transposed
message("${Yellow}---- Blocked Less Copying Transposed${ColourReset}")
add_executable("svd-blocked-less-copy-transposed" blocked_less_copy_transposed.cpp)
target_link_libraries("svd-blocked-less-copy-transposed" two-sided-svd-lib utils-lib perf-lib ${CMAKE_BINARY_DIR}/pcm/libPCM.a m)
install(TARGETS "svd-blocked-less-copy-transposed" DESTINATION ${SVD_BENCHMARK_BIN_DIR})
