add_executable(evd_perf_test evd_perf_test.cpp)

target_link_libraries(evd_perf_test gtest_main evd-lib utils-lib perf-lib m)

add_test(NAME EVD_perf COMMAND evd_perf_test)
