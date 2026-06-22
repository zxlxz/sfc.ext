aux_source_directory(. SRC)

add_library(sfc_math)
add_library(sfc::math ALIAS sfc_math)

target_sources(sfc_math PRIVATE ${SRC})
target_link_libraries(sfc_math sfc)
target_include_directories(sfc_math PUBLIC ${_SFC_EXT_INCLUDE})
