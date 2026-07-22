

include_directories(.)

add_subdirectory(sfc/math)
add_subdirectory(sfc/cuda)

target_include_directories(sfc_math PUBLIC ${CMAKE_CURRENT_LIST_DIR})
target_include_directories(sfc_cuda PUBLIC ${CMAKE_CURRENT_LIST_DIR}/sfc/math)

add_library(sfc::math ALIAS sfc_math)
add_library(sfc::cuda ALIAS sfc_cuda)
