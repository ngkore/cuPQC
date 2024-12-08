
# Commands may need to know the format version.
set(CMAKE_IMPORT_FILE_VERSION 1)

# Import target "cupqc_static" for configuration "Release"
set_property(TARGET cupqc_static APPEND PROPERTY IMPORTED_CONFIGURATIONS RELEASE)
set_target_properties(cupqc_static PROPERTIES
  IMPORTED_LINK_INTERFACE_LANGUAGES_RELEASE "CUDA"
  IMPORTED_LOCATION_RELEASE "${_IMPORT_PREFIX}/lib/libcupqc.a"
  )

list(APPEND _cmake_import_check_targets cupqc_static )
list(APPEND _cmake_import_check_files_for_cupqc_static "${_IMPORT_PREFIX}/lib/libcupqc.a" )

# Commands beyond this point should not need to know the version.
set(CMAKE_IMPORT_FILE_VERSION)
