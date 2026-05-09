aux_source_directory(. SRC)
add_library(sfc_cuda ${SRC})
target_compile_features(sfc_cuda PUBLIC cxx_std_23)
target_link_libraries(sfc_cuda PUBLIC sfc)
target_include_directories(sfc_cuda PUBLIC ${CMAKE_CURRENT_LIST_DIR}/../../)

if (APPLE)
  # MacOS does not support CUDA officially
  # but we can still build the library using the user provided CUDA headers and libraries.
  target_include_directories(sfc_cuda PUBLIC /opt/cuda/include)
  target_link_libraries(sfc_cuda PUBLIC cuda cudart cufft)
  target_link_options(sfc_cuda PUBLIC -Wl,-rpath,/usr/local/lib)
else()
  set(CUDA_ARCHITECTURES "86")
  find_package(CUDAToolkit REQUIRED)
  target_link_libraries(sfc_cuda PUBLIC CUDA::cuda_driver CUDA::cudart CUDA::cufft)
endif()
