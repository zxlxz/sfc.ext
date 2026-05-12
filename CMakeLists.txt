cmake_minimum_required(VERSION 3.21)
project(sfc.math)

add_subdirectory(../sfc/src sfc)

if (APPLE)
  include_directories(/opt/homebrew/include)
  link_directories(/opt/homebrew/lib)
endif()

add_subdirectory(src)
add_subdirectory(test)
