
list(APPEND CMAKE_PREFIX_PATH "${CMAKE_CURRENT_LIST_DIR}")

if(WIN32)
  set(_LOCAL_PREFIX "A:/.local")
  set(CMAKE_INCLUDE_PATH "${_LOCAL_PREFIX}/include")
  set(CMAKE_LIBRARY_PATH "${_LOCAL_PREFIX}/lib")
  set(CMAKE_PROGRAM_PATH "${_LOCAL_PREFIX}/bin")
endif()

if (APPLE)
  set(_HOMEBREW_PREFIX "/opt/homebrew")
  set(CMAKE_INCLUDE_PATH "${_HOMEBREW_PREFIX}/include")
  set(CMAKE_LIBRARY_PATH "${_HOMEBREW_PREFIX}/lib")
endif()

if(APPLE)
  set(CUDAToolkit_TARGET_DIR "/opt/cuda")
  set(CMAKE_CUDA_COMPILER_TOOLKIT_ROOT ${CUDAToolkit_TARGET_DIR})
  set(CUDAToolkit_INCLUDE_DIRECTORIES  ${CUDAToolkit_TARGET_DIR}/include)
endif()
