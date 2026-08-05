

add_library(_sfc_ext INTERFACE)
target_include_directories(_sfc_ext INTERFACE ${CMAKE_CURRENT_LIST_DIR})

function (add_sfc_lib NAME)
  set(_LIB sfc_${NAME})
  add_library(${_LIB})
  add_library("sfc::${NAME}" ALIAS ${_LIB})

  target_source_files(${_LIB} . "*.cc")
  target_link_libraries(${_LIB} sfc _sfc_ext)
endfunction()

function(add_sfc_test NAME)
  set(_TEST sfc-${NAME}-test)
  add_executable(${_TEST})
  target_source_files(${_TEST} . "*.cxx")
  target_link_libraries(${_TEST} sfc_${NAME})
endfunction()

function (add_sfc_mod NAME)
  add_sfc_lib(${NAME})
  add_sfc_test(${NAME})
endfunction()

# sub modules
if (CMAKE_CXX_COMPILER_ID MATCHES "Clang|GNU")
  add_compile_options(-Wall -Wextra -Wpedantic -Wconversion)
endif()

add_subdirectory(sfc/math)
add_subdirectory(sfc/cuda)
