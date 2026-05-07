add_subdirectory(sfc/cuda)
add_library(sfc::cuda ALIAS sfc_cuda)

add_subdirectory(sfc/math)
add_library(sfc::math ALIAS sfc_math)
