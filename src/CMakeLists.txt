include(../cmake/config.cmake)

set(_SFC_EXT_INCLUDE ${CMAKE_CURRENT_LIST_DIR})
if (CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
  set(_SFC_EXT_OPTIONS -Wall -Wextra -Wpedantic -Wconversion)
endif()

add_subdirectory(sfc/cuda)
add_subdirectory(sfc/math)
add_subdirectory(sfc/fft)
