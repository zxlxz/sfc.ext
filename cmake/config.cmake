
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
  )

  add_library(CUDA::cufft INTERFACE IMPORTED)
  set_target_properties(CUDA::cufft PROPERTIES
    INTERFACE_INCLUDE_DIRECTORIES "${CUDAToolkit_INCLUDE_DIRECTORIES}"
    INTERFACE_LINK_LIBRARIES "-lcufft"
  )
endif()


function(target_source_path TARGET DIR)
    set(_SOURCES)
    set(_PATTERNS ${ARGN})
    if(NOT _PATTERNS)
        set(_PATTERNS "*.cpp" "*.cxx" "*.cc" "*.cu" "*.c")
    endif()

    foreach(PATTERN ${_PATTERNS})
        file(GLOB MATCHED_SOURCES CONFIGURE_DEPENDS "${DIR}/${PATTERN}")
        list(APPEND _SOURCES ${MATCHED_SOURCES})
    endforeach()

    target_sources(${TARGET} PRIVATE ${_SOURCES})
endfunction()
