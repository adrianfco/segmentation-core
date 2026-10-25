cmake_minimum_required(VERSION 3.20)
project(segmentation_core LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)  # -std=c++17, not gnu++17

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

add_compile_options(-Wall -Wextra -Wpedantic)

# no system packages needed, both deps are fetched at configure time
include(FetchContent)

FetchContent_Declare(
  stb
  GIT_REPOSITORY https://github.com/nothings/stb.git
  GIT_TAG        2c980bb59875b0d32144a71867fbdebb2f77cd20  # pinned so wheels are reproducible
)
FetchContent_MakeAvailable(stb)

FetchContent_Declare(
  pybind11
  GIT_REPOSITORY https://github.com/pybind/pybind11.git
  GIT_TAG        v2.12.0
)
FetchContent_MakeAvailable(pybind11)

# engine is a static lib so the python module, the C++ tests and the benchmark share it
add_library(segmentation_lib STATIC
  src/image_io.cpp
  src/kmeans.cpp
  src/pfcm.cpp
  src/segmentation.cpp
)

target_include_directories(segmentation_lib PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)
# SYSTEM hides the warnings stb's headers trigger
target_include_directories(segmentation_lib SYSTEM PRIVATE ${stb_SOURCE_DIR})

# needed to link the static lib into the python .so
set_target_properties(segmentation_lib PROPERTIES POSITION_INDEPENDENT_CODE ON)

option(SEG_ENABLE_OPENMP "Parallelize the segmentation loops with OpenMP" ON)

if(SEG_ENABLE_OPENMP)
  find_package(OpenMP)
endif()

if(OpenMP_CXX_FOUND)
  target_link_libraries(segmentation_lib PUBLIC OpenMP::OpenMP_CXX)
else()
  message(STATUS "Building without OpenMP, the segmentation loops will run serially")
  # the #pragma omp lines are ignored without OpenMP, don't warn about them
  target_compile_options(segmentation_lib PRIVATE -Wno-unknown-pragmas)
endif()

pybind11_add_module(segmentation_core src/bindings.cpp)
target_link_libraries(segmentation_core PRIVATE segmentation_lib)

# top level of the wheel, so `import segmentation_core` works
install(TARGETS segmentation_core LIBRARY DESTINATION .)

add_executable(test_segmentation tests/test_segmentation.cpp)
target_link_libraries(test_segmentation PRIVATE segmentation_lib)

enable_testing()
add_test(NAME cpp_tests COMMAND test_segmentation)
# runs against the module in the build dir, not an installed copy
add_test(NAME python_tests COMMAND ${PYTHON_EXECUTABLE} -m pytest ${CMAKE_CURRENT_SOURCE_DIR}/tests)
set_tests_properties(python_tests PROPERTIES
  ENVIRONMENT PYTHONPATH=$<TARGET_FILE_DIR:segmentation_core>)

add_executable(bench_segmentation bench/bench_segmentation.cpp)
target_link_libraries(bench_segmentation PRIVATE segmentation_lib)
