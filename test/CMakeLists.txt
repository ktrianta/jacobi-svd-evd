add_executable(dummy_test dummy_test.cpp)
target_link_libraries(dummy_test gtest_main)

add_test(NAME dummy_test
         COMMAND dummy_test)
