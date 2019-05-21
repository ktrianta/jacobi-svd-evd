include(ExternalProject)

ExternalProject_Add(pcm-lib
  GIT_REPOSITORY    https://github.com/opcm/pcm.git
  GIT_TAG           master
  SOURCE_DIR        "${CMAKE_BINARY_DIR}/pcm"
  CONFIGURE_COMMAND ""
  INSTALL_COMMAND   ""
  TEST_COMMAND      ""
  UPDATE_COMMAND ""
  BUILD_IN_SOURCE 1
  BUILD_COMMAND "make"
)


set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -pthread -Wall -Wextra -std=c++11")

set(SOURCES
    tsc_x86.h
    )

add_library(perf-lib ${SOURCES})
target_include_directories(perf-lib PUBLIC . PUBLIC "${CMAKE_BINARY_DIR}/pcm")
target_link_directories(perf-lib PUBLIC "${CMAKE_BINARY_DIR}/pcm")
SET_TARGET_PROPERTIES(perf-lib PROPERTIES LINKER_LANGUAGE CXX)
