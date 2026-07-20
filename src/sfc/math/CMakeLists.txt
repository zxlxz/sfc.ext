find_package(FFTW REQUIRED)

add_library(sfc_math)
target_source_files(sfc_math . "*.cc")
target_link_libraries(sfc_math sfc)
target_include_directories(sfc_math PUBLIC ${_FFTW3f_INCLUDE})
target_include_directories(sfc_math PUBLIC ${_SFC_EXT_INCLUDE})
target_compile_options(sfc_math PRIVATE -Wall -Wextra -Wpedantic -Wconversion)

add_executable(sfc-math-test)
target_source_files(sfc-math-test . "*.cxx")
target_link_libraries(sfc-math-test PRIVATE sfc_math sfc_cuda)
