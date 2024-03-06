cmake_minimum_required(VERSION 3.10...3.26)

project(db-tetris)

# options
option(BUILD_TESTS "Build test programs" ON)
if(NOT ADJ_DELAY)
  set(ADJ_DELAY 18)
endif()
if(NOT TAP_SPEED)
  set(TAP_SPEED Tap30Hz)
endif()
if(NOT LINE_CAP)
  set(LINE_CAP 430)
endif()

if (NOT EXISTS ${CMAKE_BINARY_DIR}/CMakeCache.txt)
  if (NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE "Release" CACHE STRING "" FORCE)
  endif()
endif()

set(CMAKE_CXX_FLAGS "-std=c++20 -mbmi2 -mavx2 -mfma -Wall -Wextra -Wno-unused-parameter")
set(CMAKE_CXX_FLAGS_DEBUG "-O0 -g")
set(CMAKE_CXX_FLAGS_RELEASE "-O3 -march=native")
list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")

set(THREADS_PREFER_PTHREAD_FLAG ON)
find_package(zstd REQUIRED)
find_package(Threads REQUIRED)

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

CHECK_CXX_SOURCE_COMPILES("
    #include <immintrin.h>
    int main() {
      __m256 tmp = _mm256_set1_ps(0.0f);
      __m256 tmp2 = _mm256_fmadd_ps(tmp, tmp, tmp);
      return 0;
    }
  "
  HAVE_AVX2
)
if(NOT HAVE_AVX2)
  message(FATAL_ERROR "Must have avx2+fma support.")
endif()

include(FetchContentExclude)
FetchContent_Declare(
  argparse
  URL "https://github.com/p-ranav/argparse/archive/refs/tags/v3.0.zip"
)
FetchContent_Declare(
  spdlog
  URL "https://github.com/gabime/spdlog/archive/refs/tags/v1.12.0.zip"
)
FetchContent_Declare(
  eigen
  URL "https://gitlab.com/libeigen/eigen/-/archive/3.4.0/eigen-3.4.0.zip"
)
FetchContent_Declare(
  sparse_map
  URL "https://github.com/Tessil/sparse-map/archive/refs/tags/v0.6.2.zip"
)
FetchContent_Declare(
  hopscotch_map
  URL "https://github.com/Tessil/hopscotch-map/archive/refs/tags/v2.3.1.zip"
)
set(BOOST_INCLUDE_LIBRARIES system asio)
FetchContent_Declare(
  boost
  URL "https://github.com/boostorg/boost/releases/download/boost-1.84.0/boost-1.84.0.tar.gz"
)

FetchContent_MakeAvailable_Exclude(argparse spdlog eigen sparse_map hopscotch_map boost)

add_executable(random_boards scripts/random_boards.cpp)
add_executable(board_merge scripts/board_merge.cpp)
target_link_libraries(board_merge tsl::sparse_map)

file(GLOB MAIN_SRC "src/*.cpp")
add_executable(main ${MAIN_SRC})

target_compile_definitions(main PRIVATE -DTAP_SPEED=${TAP_SPEED} -DADJ_DELAY=${ADJ_DELAY} -DLINE_CAP=${LINE_CAP})
target_link_libraries(main argparse::argparse spdlog::spdlog zstd eigen tsl::sparse_map tsl::hopscotch_map Boost::asio)

if(BUILD_TESTS)
  enable_testing()
  FetchContent_Declare(
    googletest
    URL "https://github.com/google/googletest/archive/refs/tags/v1.14.0.zip"
  )
  set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
  FetchContent_MakeAvailable_Exclude(googletest)

  file(GLOB TEST_SRC "test/*.cpp" "src/files.cpp" "src/config.cpp")
  add_executable(run-test ${TEST_SRC})
  target_link_libraries(run-test gtest_main zstd tsl::sparse_map tsl::hopscotch_map)
  target_compile_definitions(run-test PRIVATE -DTESTING=1)

  include(GoogleTest)
  gtest_discover_tests(run-test)
endif()
