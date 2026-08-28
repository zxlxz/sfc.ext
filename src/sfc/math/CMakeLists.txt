
add_sfc_mod(math)

find_package(FFTW REQUIRED)
target_include_directories(sfc_math PUBLIC ${_FFTW3f_INCLUDE})
