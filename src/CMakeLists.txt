include(../cmake/config.cmake)

add_library(_sfc_ext INTERFACE)
target_link_libraries(_sfc_ext INTERFACE sfc)
target_include_directories(_sfc_ext INTERFACE ${CMAKE_CURRENT_LIST_DIR})

add_subdirectory(sfc/cuda)
add_library(sfc::cuda ALIAS sfc_cuda)

add_subdirectory(sfc/math)
add_library(sfc::math ALIAS sfc_math)
