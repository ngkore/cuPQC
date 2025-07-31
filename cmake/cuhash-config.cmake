#cuhash-config.cmake
#
# Imported interface targets provided:
#  * ::_static - static library target
#  * :: - alias to static library target
#

set(cuhash_VERSION "0.3.0")
# build: 


####### Expanded from @PACKAGE_INIT@ by configure_package_config_file() #######
####### Any changes to this file will be overwritten by the next CMake run ####
####### The input file was cuhash-config.cmake.in                            ########

get_filename_component(PACKAGE_PREFIX_DIR "${CMAKE_CURRENT_LIST_DIR}/../../../" ABSOLUTE)

macro(set_and_check _var _file)
  set(${_var} "${_file}")
  if(NOT EXISTS "${_file}")
    message(FATAL_ERROR "File or directory ${_file} referenced by variable ${_var} does not exist !")
  endif()
endmacro()

macro(check_required_components _NAME)
  foreach(comp ${${_NAME}_FIND_COMPONENTS})
    if(NOT ${_NAME}_${comp}_FOUND)
      if(${_NAME}_FIND_REQUIRED_${comp})
        set(${_NAME}_FOUND FALSE)
      endif()
    endif()
  endforeach()
endmacro()

####################################################################################

include("${CMAKE_CURRENT_LIST_DIR}/cuhash-static-targets.cmake" OPTIONAL)

set(cuhash_STATIC_LIBRARIES cuhash_static)

# Allow Alias to targets
set_target_properties(${cuhash_STATIC_LIBRARIES} PROPERTIES IMPORTED_GLOBAL 1)

add_library(cuhash ALIAS cuhash_static)
add_library(:: ALIAS cuhash_static)
add_library(::_static ALIAS cuhash_static)

get_target_property(cuhash_LOCATION ${cuhash_STATIC_LIBRARIES} LOCATION)
get_filename_component(cuhash_LOCATION ${cuhash_LOCATION} DIRECTORY)
get_filename_component(cuhash_LOCATION ${cuhash_LOCATION}/.. REALPATH)

check_required_components(cuhash)
message(STATUS "Found cuhash: (Location: ${cuhash_LOCATION} Version: ${cuhash_VERSION}")
