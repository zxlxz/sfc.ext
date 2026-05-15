aux_source_directory(. SRC)
add_library(sfc_math ${SRC})

find_package(FFTW)
target_link_libraries(sfc_math PUBLIC sfc_cuda)
target_link_libraries(sfc_math PUBLIC fftw3f)
