
project(cuda)

if(APPLE)
  find_path(_CUDA_INCLUDE cuda_runtime_api.h PATHS /opt/cuda/include REQUIRED)
  get_filename_component(_CUDA_PATH "${_CUDA_INCLUDE}" DIRECTORY)
  message(STATUS "CUDA_PATH: ${_CUDA_PATH}")
  message(STATUS "CUDA_INCLUDE_DIRS: ${_CUDA_INCLUDE}")

  set(CUDAToolkit_TARGET_DIR ${_CUDA_PATH} DIRECTORY)
  set(CMAKE_CUDA_COMPILER_TOOLKIT_ROOT ${_CUDA_PATH})
  set(CUDAToolkit_INCLUDE_DIRECTORIES "${_CUDA_PATH}/include")

  find_library(_CUDA_LIBRARY "cuda" PATHS ${_CUDA_PATH}/lib REQUIRED)
  message(STATUS "CUDA_LIBRARY: ${_CUDA_LIBRARY}")
  add_library(CUDA::cuda_driver INTERFACE IMPORTED)
  set_target_properties(CUDA::cuda_driver PROPERTIES
    INTERFACE_INCLUDE_DIRECTORIES "${CUDAToolkit_INCLUDE_DIRECTORIES}"
    INTERFACE_LINK_LIBRARIES "${_CUDA_LIBRARY}"
  )

  find_library(_CUDART_LIBRARY "cudart" PATHS ${_CUDA_PATH}/lib REQUIRED)
  message(STATUS "CUDART_LIBRARY: ${_CUDART_LIBRARY}")
  add_library(CUDA::cudart INTERFACE IMPORTED)
  set_target_properties(CUDA::cudart PROPERTIES
    INTERFACE_INCLUDE_DIRECTORIES "${CUDAToolkit_INCLUDE_DIRECTORIES}"
    INTERFACE_LINK_LIBRARIES "${_CUDART_LIBRARY}"
  )

  find_library(_CUFFT_LIBRARY "cufft" PATHS ${_CUDA_PATH}/lib REQUIRED)
  message(STATUS "CUFFT_LIBRARY: ${_CUFFT_LIBRARY}")
  add_library(CUDA::cufft INTERFACE IMPORTED)
  target_link_libraries(CUDA::cufft INTERFACE CUDA::cudart)
  set_target_properties(CUDA::cufft PROPERTIES
    INTERFACE_LINK_LIBRARIES "${_CUFFT_LIBRARY}"
  )

endif()
