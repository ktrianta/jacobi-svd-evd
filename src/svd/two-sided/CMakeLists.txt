set(SOURCES
    svd.cpp
    nsvd.cpp
    )

add_library(two-sided-svd-lib STATIC ${SOURCES})
target_include_directories(two-sided-svd-lib PUBLIC .)
target_link_libraries(two-sided-svd-lib utils-lib)

add_executable(two-sided-svd main.cpp)
target_link_libraries(two-sided-svd utils-lib m)
