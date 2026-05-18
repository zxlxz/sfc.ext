
find_path(FFTW_INCLUDE fftw3.h)
find_library(FFTW_LIBRARY fftw3f)

add_library(FFTW::fftw3f SHARED IMPORTED)
set_target_properties(FFTW::fftw3f PROPERTIES
  INTERFACE_INCLUDE_DIRECTORIES "${FFTW_INCLUDE}"
  IMPORTED_IMPLIB "${FFTW_LIBRARY}"
)

if(WIN32)
  find_program(FFTW_DLL libfftw3f-3.dll)
  set_target_properties(FFTW::fftw3f PROPERTIES
    IMPORTED_LOCATION "${FFTW_DLL}"
  )
endif()
