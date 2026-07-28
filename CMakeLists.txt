cmake_minimum_required(VERSION 3.21)

project(sfc.ext)

include(cmake/config.cmake)

add_subdirectory(../sfc ../../sfc/build)
add_subdirectory(src)
