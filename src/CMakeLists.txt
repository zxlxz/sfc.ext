include(../cmake/config.cmake)
target_include_directories(_sfc_ext INTERFACE ${CMAKE_CURRENT_LIST_DIR})

add_subdirectory(sfc/cuda)
add_subdirectory(sfc/math)
add_subdirectory(sfc/fft)
