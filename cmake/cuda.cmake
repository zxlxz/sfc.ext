
project(cuda)

if(APPLE)
  find_path(CUDAToolkit_INCLUDE_DIRECTORIES "cuda_runtime_api.h"
    PATHS
      "/opt/cuda/include"
      "/usr/local/cuda/include"
      "/usr/local/include"
  )

  get_filename_component(CUDAToolkit_TARGET_DIR "${CUDAToolkit_INCLUDE_DIRECTORIES}" DIRECTORY)
  message(STATUS "CUDA Toolkit Directory: ${CUDAToolkit_TARGET_DIR}")
  set(CMAKE_CUDA_COMPILER_TOOLKIT_ROOT ${CUDAToolkit_TARGET_DIR})

  find_library(_CUDART_LIBRARY "cudart" PATHS ${CUDAToolkit_TARGET_DIR}/lib)
  add_library(CUDA::cudart INTERFACE IMPORTED)
  set_target_properties(CUDA::cudart PROPERTIES
    INTERFACE_INCLUDE_DIRECTORIES "${CUDAToolkit_INCLUDE_DIRECTORIES}"
    INTERFACE_LINK_LIBRARIES "${_CUDART_LIBRARY}"
  )

  find_library(_CUFFT_LIBRARY "cufft" PATHS ${CUDAToolkit_TARGET_DIR}/lib)
  add_library(CUDA::cufft INTERFACE IMPORTED)
  set_target_properties(CUDA::cufft PROPERTIES
    INTERFACE_LINK_LIBRARIES "${_CUFFT_LIBRARY}"
  )
  target_link_libraries(CUDA::cufft INTERFACE CUDA::cudart)
endif()
