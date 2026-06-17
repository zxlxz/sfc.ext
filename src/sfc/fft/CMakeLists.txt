find_package(FFTW REQUIRED)
find_package(CUDAToolkit REQUIRED)

aux_source_directory(. SRC)
add_library(sfc_fft)
add_library(sfc::fft ALIAS sfc_fft)
target_sources(sfc_fft PRIVATE ${SRC})
target_compile_options(sfc_fft PRIVATE ${_SFC_EXT_OPTIONS})

target_link_libraries(sfc_fft PUBLIC sfc FFTW::fftw3f CUDA::cufft)
target_include_directories(sfc_fft PUBLIC ${_SFC_EXT_INCLUDE})
