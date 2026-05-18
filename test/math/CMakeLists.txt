aux_source_directory(. SRC)
add_executable(sfc_math_test ${SRC})
target_link_libraries(sfc_math_test sfc_math)
