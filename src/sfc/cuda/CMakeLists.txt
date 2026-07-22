if (NOT APPLE)
  find_package(CUDAToolkit REQUIRED)
endif()

add_library(sfc_cuda)
target_source_files(sfc_cuda . "*.cc")
target_link_libraries(sfc_cuda PUBLIC sfc CUDA::cudart)

add_executable(sfc-cuda-test)
target_source_files(sfc-cuda-test . "*.cxx")
target_link_libraries(sfc-cuda-test PRIVATE sfc_cuda sfc_math)
