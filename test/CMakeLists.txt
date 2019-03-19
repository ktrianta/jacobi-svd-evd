set(SRC_FILES
    ../src/util.cpp
    ../src/evd.cpp
    ../src/svd.cpp)

add_executable(tests svd_test.cpp evd_test.cpp util_test.cpp ${SRC_FILES})
target_link_libraries(tests gtest_main)
