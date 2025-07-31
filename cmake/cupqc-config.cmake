#cupqc-config.cmake
#
# Imported interface targets provided:
#  * ::_static - static library target
#  * :: - alias to static library target
#

set(cupqc_VERSION "0.3.0")
# build: 


####### Expanded from @PACKAGE_INIT@ by configure_package_config_file() #######
####### Any changes to this file will be overwritten by the next CMake run ####
####### The input file was cupqc-config.cmake.in                            ########

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

include("${CMAKE_CURRENT_LIST_DIR}/cupqc-static-targets.cmake" OPTIONAL)

set(cupqc_STATIC_LIBRARIES cupqc_static)

# Allow Alias to targets
set_target_properties(${cupqc_STATIC_LIBRARIES} PROPERTIES IMPORTED_GLOBAL 1)

add_library(cupqc ALIAS cupqc_static)
add_library(:: ALIAS cupqc_static)
add_library(::_static ALIAS cupqc_static)

get_target_property(cupqc_LOCATION ${cupqc_STATIC_LIBRARIES} LOCATION)
get_filename_component(cupqc_LOCATION ${cupqc_LOCATION} DIRECTORY)
get_filename_component(cupqc_LOCATION ${cupqc_LOCATION}/.. REALPATH)

check_required_components(cupqc)
message(STATUS "Found cupqc: (Location: ${cupqc_LOCATION} Version: ${cupqc_VERSION}")
