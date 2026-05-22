aux_source_directory(. SRC)
add_library(sfc_cuda ${SRC})
target_link_libraries(sfc_cuda PUBLIC _sfc_ext)

set(CUDA_ARCHITECTURES "80")
find_package(CUDAToolkit REQUIRED)
target_link_libraries(sfc_cuda PUBLIC CUDA::cuda_driver CUDA::cudart CUDA::cufft)
