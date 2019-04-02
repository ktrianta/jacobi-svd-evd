add_executable(svd_test svd_test.cpp)
target_link_libraries(svd_test gtest_main svd-lib utils-lib m)

add_test(NAME SVD COMMAND svd_test)
