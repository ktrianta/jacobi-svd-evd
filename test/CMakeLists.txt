set(SRC_FILES
    ../src/util.cpp
    ../src/svd.cpp)

add_executable(tests util_test.cpp ${SRC_FILES})
target_link_libraries(tests gtest_main)
