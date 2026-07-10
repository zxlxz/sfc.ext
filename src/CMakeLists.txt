include(../cmake/config.cmake)

set(_SFC_EXT_INCLUDE ${CMAKE_CURRENT_LIST_DIR})

add_subdirectory(sfc/math)
add_subdirectory(sfc/cuda)

add_library(sfc::math ALIAS sfc_math)
add_library(sfc::cuda ALIAS sfc_cuda)
