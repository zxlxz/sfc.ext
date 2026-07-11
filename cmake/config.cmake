
list(APPEND CMAKE_PREFIX_PATH "${CMAKE_CURRENT_LIST_DIR}")

if(WIN32)
  set(_LOCAL_PREFIX "A:/.local")
  set(CMAKE_INCLUDE_PATH "${_LOCAL_PREFIX}/include")
  set(CMAKE_LIBRARY_PATH "${_LOCAL_PREFIX}/lib")
  set(CMAKE_PROGRAM_PATH "${_LOCAL_PREFIX}/bin")
endif()

# fake cuda toolkit for Apple Silicon
if(APPLE)
  set(CUDAToolkit_TARGET_DIR "/usr/local")
  set(CMAKE_CUDA_COMPILER_TOOLKIT_ROOT ${CUDAToolkit_TARGET_DIR})
  set(CUDAToolkit_INCLUDE_DIRECTORIES  ${CUDAToolkit_TARGET_DIR}/include)

  add_library(CUDA::cudart INTERFACE IMPORTED)
  set_target_properties(CUDA::cudart PROPERTIES
    INTERFACE_INCLUDE_DIRECTORIES "${CUDAToolkit_INCLUDE_DIRECTORIES}"
    INTERFACE_LINK_LIBRARIES "-lcudart"
    INTERFACE_LINK_OPTIONS "LINKER:-rpath,${CUDAToolkit_TARGET_DIR}/lib"
  )

  add_library(CUDA::cufft INTERFACE IMPORTED)
  target_link_libraries(CUDA::cufft INTERFACE CUDA::cudart)
  set_target_properties(CUDA::cufft PROPERTIES
    INTERFACE_LINK_LIBRARIES "-lcufft"
  )
endif()
