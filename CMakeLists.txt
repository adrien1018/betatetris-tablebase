cmake_minimum_required(VERSION 3.10)

project(db-tetris)

set(CMAKE_CXX_FLAGS "-O3 -march=native -Wall -Wextra -Wno-unused-parameter")

set(THREADS_PREFER_PTHREAD_FLAG ON)
find_package(Threads REQUIRED)
find_library(TBB_LIB tbb)

include(TestBigEndian)
TEST_BIG_ENDIAN(IS_BIG_ENDIAN)
if(IS_BIG_ENDIAN)
  message(FATAL_ERROR "Must be little endian.")
endif()

include(CheckCXXSourceCompiles)
CHECK_CXX_SOURCE_COMPILES("
    #include <immintrin.h>
    int main() {
      int tmp = _pext_u64(0,0);
      int tmp2 = _pdep_u64(0,0);
      return 0;
    }
  "
  HAVE_BMI2
)
if(NOT HAVE_BMI2)
  message(FATAL_ERROR "Must have bmi2 support.")
endif()

add_executable(graph_gen graph_gen.cpp)
add_executable(gametree gametree.cpp)

target_link_libraries(graph_gen PRIVATE Threads::Threads "${TBB_LIB}")
target_link_libraries(gametree PRIVATE Threads::Threads)
