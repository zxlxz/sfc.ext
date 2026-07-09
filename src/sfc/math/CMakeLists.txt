find_package(FFTW REQUIRED)

add_library(sfc_math)
target_source_path(sfc_math . "*.cc")
target_link_libraries(sfc_math sfc FFTW::fftw3f)
target_include_directories(sfc_math PUBLIC ${_SFC_EXT_INCLUDE})
target_compile_options(sfc_math PRIVATE -Wall -Wextra -Wpedantic -Wconversion)

add_executable(sfc_math_test)
target_source_path(sfc_math_test . "*.cxx")
target_link_libraries(sfc_math_test PRIVATE sfc_math sfc_cuda)
