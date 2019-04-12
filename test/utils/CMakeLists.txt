cmake_minimum_required(VERSION 3.9.2)
add_executable(util_test util_test.cpp)
target_link_libraries(util_test gtest_main utils-lib m)

add_test(NAME Utilities COMMAND util_test)
