set(SRC_FILES
    ../src/util.cpp
    ../src/svd.cpp)

add_executable(svd_test svd_test.cpp ${SRC_FILES})
add_executable(util_test util_test.cpp ${SRC_FILES})
target_link_libraries(svd_test gtest_main)
target_link_libraries(util_test gtest_main)

add_test(NAME SVD COMMAND svd_test)
add_test(NAME Utilities COMMAND util_test)
