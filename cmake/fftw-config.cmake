

find_path(_FFTW3f_INCLUDE fftw3.h)
find_library(_FFTW3f_LIBRARY fftw3f)

add_library(FFTW::fftw3f UNKNOWN IMPORTED)
set_target_properties(FFTW::fftw3f PROPERTIES
  INTERFACE_INCLUDE_DIRECTORIES "${_FFTW3f_INCLUDE}"
  IMPORTED_LOCATION "${_FFTW3f_LIBRARY}"
)

if(WIN32)
  find_program(_FFTW3f_DLL "libfftw3f-3.dll")
  set_target_properties(FFTW::fftw3f PROPERTIES
    IMPORTED_IMPLIB "${_FFTW3f_DLL}"
  )
endif()
