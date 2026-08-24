
add_sfc_mod(math)
target_link_libraries(sfc_math sfc_cuda)

find_package(FFTW REQUIRED)
target_include_directories(sfc_math PUBLIC ${_FFTW3f_INCLUDE})
