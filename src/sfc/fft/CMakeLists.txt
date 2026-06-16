aux_source_directory(. SRC)
add_library(sfc_fft ${SRC})

find_package(FFTW REQUIRED)
find_package(CUDAToolkit REQUIRED)

target_link_libraries(sfc_fft PUBLIC sfc_math sfc_cuda)
target_link_libraries(sfc_fft PUBLIC FFTW::fftw3f CUDA::cufft)
add_library(sfc::fft ALIAS sfc_fft)
