if (NOT APPLE)
  find_package(CUDAToolkit REQUIRED)
endif()

add_library(sfc_cuda)
target_source_files(sfc_cuda . "*.cc")
target_link_libraries(sfc_cuda PUBLIC sfc)
target_link_libraries(sfc_cuda PUBLIC CUDA::cudart CUDA::cufft)
target_include_directories(sfc_cuda PUBLIC ${_SFC_EXT_INCLUDE})
target_compile_options(sfc_cuda PRIVATE -Wall -Wextra -Wpedantic -Wconversion)

add_executable(sfc-cuda-test)
target_source_files(sfc-cuda-test . "*.cxx")
target_link_libraries(sfc-cuda-test PRIVATE sfc_cuda sfc_math)
