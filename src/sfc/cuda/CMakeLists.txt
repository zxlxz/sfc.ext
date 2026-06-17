
set(CUDA_ARCHITECTURES "80")
find_package(CUDAToolkit REQUIRED)

aux_source_directory(. SRC)

add_library(sfc_cuda)
add_library(sfc::cuda ALIAS sfc_cuda)
target_sources(sfc_cuda PRIVATE ${SRC})
target_compile_options(sfc_cuda PRIVATE ${_SFC_EXT_OPTIONS})

target_link_libraries(sfc_cuda PUBLIC sfc CUDA::cudart)
target_include_directories(sfc_cuda PUBLIC ${_SFC_EXT_INCLUDE})
