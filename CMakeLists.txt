cmake_minimum_required(VERSION 3.15)
project(segmentation_core VERSION 0.1.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Build type default
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

# Compiler warnings
if(MSVC)
  add_compile_options(/W4)
else()
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

# -------------------------
# Dependencies
# -------------------------

# stb_image (header-only, vendored or fetched)
include(FetchContent)

FetchContent_Declare(
  stb
  GIT_REPOSITORY https://github.com/nothings/stb.git
  GIT_TAG        master
)
FetchContent_MakeAvailable(stb)

# pybind11
FetchContent_Declare(
  pybind11
  GIT_REPOSITORY https://github.com/pybind/pybind11.git
  GIT_TAG        v2.12.0
)
FetchContent_MakeAvailable(pybind11)

# -------------------------
# Core library
# -------------------------

add_library(segmentation_lib STATIC
  src/image_io.cpp
  src/kmeans.cpp
  src/pfcm.cpp
  src/segmentation.cpp
)

target_include_directories(segmentation_lib PUBLIC
  ${CMAKE_CURRENT_SOURCE_DIR}/include
  ${stb_SOURCE_DIR}
)

set_target_properties(segmentation_lib PROPERTIES POSITION_INDEPENDENT_CODE ON)

# -------------------------
# Python module
# -------------------------

pybind11_add_module(segmentation_core
  src/bindings.cpp
)

target_link_libraries(segmentation_core PRIVATE segmentation_lib)

# -------------------------
# C++ tests
# -------------------------

enable_testing()

add_executable(test_segmentation
  tests/test_segmentation.cpp
)

target_link_libraries(test_segmentation PRIVATE segmentation_lib)

add_test(NAME cpp_tests COMMAND test_segmentation)

# -------------------------
# Install
# -------------------------

install(TARGETS segmentation_core LIBRARY DESTINATION .)
