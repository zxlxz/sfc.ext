aux_source_directory(. SRC)
add_library(sfc_math ${SRC})

target_link_libraries(sfc_math PUBLIC sfc_cuda)

find_package(FFTW REQUIRED)
target_link_libraries(sfc_math PUBLIC FFTW::fftw3f)
