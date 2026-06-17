aux_source_directory(. SRC)

add_library(sfc_math INTERFACE)
add_library(sfc::math ALIAS sfc_math)

target_link_libraries(sfc_math INTERFACE sfc)
target_include_directories(sfc_math INTERFACE ${_SFC_EXT_INCLUDE})
