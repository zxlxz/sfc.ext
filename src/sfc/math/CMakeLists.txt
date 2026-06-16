aux_source_directory(. SRC)
add_library(sfc_math INTERFACE)
target_link_libraries(sfc_math INTERFACE _sfc_ext)
add_library(sfc::math ALIAS sfc_math)
