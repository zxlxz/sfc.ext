cmake_minimum_required(VERSION 3.21)
project(sfc.math)

add_subdirectory(../sfc/src sfc)

if (APPLE)
  include_directories(/opt/homebrew/include)
  link_directories(/opt/homebrew/lib)
endif()

if (EXISTS ${CMAKE_CURRENT_LIST_DIR}/3rdparty/fftw)
  add_subdirectory(3rdparty/fftw)
endif()

add_subdirectory(src)
add_subdirectory(test)
