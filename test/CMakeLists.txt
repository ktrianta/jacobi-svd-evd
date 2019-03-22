set(SOURCES
    ../src/util.cpp
    ../src/svd.cpp
    )

add_library(sources_lib STATIC ${SOURCES})

add_executable(svd_test svd_test.cpp)
add_executable(util_test util_test.cpp)
target_link_libraries(svd_test gtest_main sources_lib)
target_link_libraries(util_test gtest_main sources_lib)

add_test(NAME SVD COMMAND svd_test)
add_test(NAME Utilities COMMAND util_test)