

include_directories(.)
add_compile_options(-Wall -Wextra -Wpedantic -Wconversion)

set(_SFC_EXT_PATH ${CMAKE_CURRENT_LIST_DIR})

function (add_sfc_mod NAME)
  set(_LIB sfc_${NAME})
  set(_TEST sfc-${NAME}-test)
  add_library(${_LIB})
  add_library("sfc::${NAME}" ALIAS ${_LIB})

  target_source_files(${_LIB} . "*.cc")
  target_link_libraries(${_LIB} sfc)
  target_include_directories(${_LIB} PUBLIC ${_SFC_EXT_PATH})

  add_executable(${_TEST})
  target_source_files(${_TEST} . "*.cxx")
  target_link_libraries(${_TEST} ${_LIB})
endfunction()

add_subdirectory(sfc/math)
add_subdirectory(sfc/cuda)
add_subdirectory(sfc/dcm)
