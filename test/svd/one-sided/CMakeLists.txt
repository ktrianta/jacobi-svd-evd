cmake_minimum_required(VERSION 3.12)
add_executable(svd_test svd_test.cpp ../../test_utils.cpp)
target_link_libraries(svd_test gtest_main svd-lib utils-lib m)

add_test(NAME SVD COMMAND svd_test WORKING_DIRECTORY ${CMAKE_SOURCE_DIR})
