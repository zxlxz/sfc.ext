cmake_minimum_required(VERSION 3.21)
project(sfc.math)

add_subdirectory(../sfc/src sfc)

include(cmake/config.cmake)

add_subdirectory(src)
add_subdirectory(test)
