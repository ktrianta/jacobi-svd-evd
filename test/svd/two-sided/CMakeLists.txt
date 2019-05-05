add_executable(two-sided-svd-test svd_test.cpp svd_blocked_test.cpp ../../test_utils.cpp)
target_link_libraries(two-sided-svd-test gtest_main two-sided-svd-lib utils-lib m)

add_test(NAME Two-Sided-SVD COMMAND two-sided-svd-test WORKING_DIRECTORY ${CMAKE_SOURCE_DIR})
