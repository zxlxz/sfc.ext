if (NOT APPLE)
  find_package(CUDAToolkit REQUIRED)
endif()

add_sfc_mod(cuda)
target_link_libraries(sfc_cuda CUDA::cudart)

target_link_libraries(sfc-cuda-test sfc_math)
