
if (NOT APPLE)
  find_package(CUDAToolkit REQUIRED)
endif()

add_library(sfc_cuda)
add_library(sfc::cuda ALIAS sfc_cuda)
target_source_path(sfc_cuda .)
target_compile_options(sfc_cuda PRIVATE ${_SFC_EXT_OPTIONS})
target_link_libraries(sfc_cuda PUBLIC sfc CUDA::cudart)
target_include_directories(sfc_cuda PUBLIC ${_SFC_EXT_INCLUDE})
