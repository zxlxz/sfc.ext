cmake_minimum_required(VERSION 3.21)

if (CMAKE_CXX_COMPILER_ID STREQUAL "GNU" OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic -Wconversion)
endif()

# sub tests
add_subdirectory(cuda)
add_subdirectory(math)
