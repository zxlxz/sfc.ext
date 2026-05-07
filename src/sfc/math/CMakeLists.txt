aux_source_directory(. SRC)
add_library(sfc_math ${SRC})

target_compile_features(sfc_math PUBLIC cxx_std_23)
target_include_directories(sfc_math PUBLIC ${CMAKE_CURRENT_LIST_DIR}/../../)

target_link_libraries(sfc_math PUBLIC sfc_cuda)
