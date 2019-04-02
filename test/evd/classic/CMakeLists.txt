add_executable(evd_test evd_test.cpp)
target_link_libraries(evd_test gtest_main evd-lib utils-lib m)

add_test(NAME EVD COMMAND evd_test)
