cmake_minimum_required(VERSION 3.21)
project(sfc.math)

add_subdirectory(../sfc/src sfc)

if (CMAKE_CXX_COMPILER_ID MATCHES "Clang" OR CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
  add_compile_options(-Wall -Wmost -Wextra -Wpedantic -Wconversion)
endif()

add_subdirectory(src)
add_subdirectory(test)
