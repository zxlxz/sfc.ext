
if (APPLE)
  set(_HOMEBREW_DIR "/opt/homebrew")
  list(APPEND CMAKE_INCLUDE_PATH "${_HOMEBREW_DIR}/include")
  list(APPEND CMAKE_LIBRARY_PATH "${_HOMEBREW_DIR}/lib")
endif()
