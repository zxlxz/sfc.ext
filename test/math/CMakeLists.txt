aux_source_directory(. SRC)
add_executable(sfc_math_test ${SRC})

target_link_libraries(sfc_math_test sfc_math)

if(WIN32)
  find_package(FFTW)

  add_custom_command(TARGET sfc_math_test POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
       "$<TARGET_PROPERTY:fftw3f,IMPORTED_LOCATION>"
       "$<TARGET_FILE_DIR:sfc_math_test>")
endif()
