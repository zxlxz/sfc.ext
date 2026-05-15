cmake_minimum_required(VERSION 3.21)
project(sfc.math)
list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_LIST_DIR}/3rdparty/cmake")

add_subdirectory(../sfc/src sfc)

include(cmake/config.cmake)
add_subdirectory(src)
add_subdirectory(test)
