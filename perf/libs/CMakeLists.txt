set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Wextra -std=c++11")

set(SOURCES
    tsc_x86.h
    )

add_library(perf-lib ${SOURCES})
target_include_directories(perf-lib PUBLIC .)
SET_TARGET_PROPERTIES(perf-lib PROPERTIES LINKER_LANGUAGE CXX)
