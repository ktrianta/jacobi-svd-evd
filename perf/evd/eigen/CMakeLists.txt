add_executable(eigen_evd_perf evd_perf.cpp)
target_link_libraries(eigen_evd_perf utils-lib perf-lib ${CMAKE_BINARY_DIR}/pcm/libPCM.a m)
target_include_directories(eigen_evd_perf PRIVATE ${EIGEN_LIB_DIR})
install(TARGETS eigen_evd_perf DESTINATION ${EVD_BENCHMARK_BIN_DIR})
