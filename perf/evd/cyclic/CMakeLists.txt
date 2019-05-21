#add_executable(evd_cyclic_perf evd_cyclic_perf.cpp)
#target_link_libraries(evd_cyclic_perf evd-cyclic-lib utils-lib perf-lib ${CMAKE_BINARY_DIR}/pcm/libPCM.a m)
#install(TARGETS evd_cyclic_perf DESTINATION ${EVD_BENCHMARK_BIN_DIR})

add_executable(evd_cyclic_vectorize_perf evd_cyclic_vectorize_perf.cpp)
target_link_libraries(evd_cyclic_vectorize_perf evd-cyclic-lib utils-lib perf-lib ${CMAKE_BINARY_DIR}/pcm/libPCM.a m)
install(TARGETS evd_cyclic_vectorize_perf DESTINATION ${EVD_BENCHMARK_BIN_DIR})


add_executable(evd_cyclic_perf_org evd_cyclic_perf_org.cpp)
target_link_libraries(evd_cyclic_perf_org evd-cyclic-lib utils-lib perf-lib ${CMAKE_BINARY_DIR}/pcm/libPCM.a m)
install(TARGETS evd_cyclic_perf_org DESTINATION ${EVD_BENCHMARK_BIN_DIR})

add_executable(evd_cyclic_vectorize_raj_perf evd_cyclic_vectorize_raj_perf.cpp)
target_link_libraries(evd_cyclic_vectorize_raj_perf evd-cyclic-lib utils-lib perf-lib ${CMAKE_BINARY_DIR}/pcm/libPCM.a m)
install(TARGETS evd_cyclic_vectorize_raj_perf DESTINATION ${EVD_BENCHMARK_BIN_DIR})
