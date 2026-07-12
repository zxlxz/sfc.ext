cmake_minimum_required(VERSION 3.21)
project(sfc.ext)

include(cmake/config.cmake)
include(cmake/cuda.cmake)

add_subdirectory(../sfc sfc)
add_subdirectory(src)
