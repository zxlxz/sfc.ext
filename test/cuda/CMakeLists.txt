aux_source_directory(. SRC)
add_executable(sfc_cuda_test ${SRC})

target_link_libraries(sfc_cuda_test sfc sfc_cuda)
