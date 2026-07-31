cmake_minimum_required(VERSION 3.16 FATAL_ERROR)
project(PhLiNO)

include(CTest)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

set(CMAKE_CXX_STANDARD 23)

include(cmake/mpi.cmake)

add_subdirectory(external/eigen)
set(EIGEN3_INCLUDE_DIR "${CMAKE_CURRENT_LIST_DIR}/external/eigen/")
include_directories(${EIGEN3_INCLUDE_DIR})

find_package(ROOT REQUIRED COMPONENTS Minuit2 Minuit)

# Boost.System is header only since 1.69 and is no longer shipped as a separate component.
find_package(Boost COMPONENTS filesystem iostreams program_options log timer REQUIRED)

# Find OpenMP. AppleClang ships without the runtime and without the headers on the
# default search paths, so the plain find_package() probe fails even when Homebrew's
# libomp is installed. Feed it the brew prefix and probe again.
find_package(OpenMP)
if(NOT OpenMP_CXX_FOUND AND APPLE)
  execute_process(COMMAND brew --prefix libomp
                  OUTPUT_VARIABLE LIBOMP_PREFIX
                  OUTPUT_STRIP_TRAILING_WHITESPACE
                  ERROR_QUIET)
  if(LIBOMP_PREFIX AND EXISTS "${LIBOMP_PREFIX}/lib/libomp.dylib")
    set(OpenMP_C_FLAGS "-Xclang -fopenmp -I${LIBOMP_PREFIX}/include")
    set(OpenMP_CXX_FLAGS "-Xclang -fopenmp -I${LIBOMP_PREFIX}/include")
    set(OpenMP_C_LIB_NAMES omp)
    set(OpenMP_CXX_LIB_NAMES omp)
    set(OpenMP_omp_LIBRARY "${LIBOMP_PREFIX}/lib/libomp.dylib")
    find_package(OpenMP)
  endif()
endif()
find_package(Arrow REQUIRED CONFIG)
find_package(Parquet REQUIRED CONFIG)
# CONFIG mode (not the CMake-bundled FindProtobuf.cmake module) so the
# protobuf::libprotobuf target carries its real transitive link deps -- modern
# protobuf builds against abseil, and the module-mode target does not
# propagate those abseil libraries, which fails at link time.
find_package(Protobuf REQUIRED CONFIG)

include(FetchContent)
FetchContent_Declare(json URL https://github.com/nlohmann/json/releases/download/v3.11.3/json.tar.xz)
FetchContent_MakeAvailable(json)

if(BUILD_TESTING)
  set(BUILD_GMOCK OFF CACHE BOOL "" FORCE)
  set(INSTALL_GTEST OFF CACHE BOOL "" FORCE)
  if(MSVC)
    set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
  endif()

  FetchContent_Declare(
    googletest
    URL https://github.com/google/googletest/archive/52eb8108c5bdec04579160ae17225d66034bd723.zip
  )
  FetchContent_MakeAvailable(googletest)
endif()

# The OpenMP::OpenMP_CXX imported target carries both the compile flag and the
# runtime library. The old CMAKE_*_FLAGS route relied on OpenMP_EXE_LINKER_FLAGS,
# which modern FindOpenMP no longer sets, so the pragmas would compile but never
# link. Targets that contain omp pragmas link the imported target instead.
if(OpenMP_CXX_FOUND)
  message(STATUS "OpenMP enabled: ${OpenMP_CXX_FLAGS}")
else()
  message(STATUS "OpenMP not found -- omp pragmas will be ignored")
endif()

macro(remove_flag_from_file _target _file _flag)
  get_target_property(_target_sources ${_target} SOURCES)
  # Check if a sync is required, in which case we'll force a rewrite of the cache variables.
  if(_flag_sync_required)
    unset(_cached_${_target}_cxx_flags CACHE)
    unset(_cached_${_target}_${_file}_cxx_flags CACHE)
  endif()
  get_target_property(_${_target}_cxx_flags ${_target} COMPILE_OPTIONS)
  # On first entry, cache the target compile flags and apply them to each source file
  # in the target.
  if(NOT _cached_${_target}_cxx_flags)
    # Obtain and cache the target compiler options, then clear them.
    get_target_property(_target_cxx_flags ${_target} COMPILE_OPTIONS)
    set(_cached_${_target}_cxx_flags "${_target_cxx_flags}" CACHE INTERNAL "")
    set_target_properties(${_target} PROPERTIES COMPILE_OPTIONS "")
    # Apply the target compile flags to each source file.
    foreach(_source_file ${_target_sources})
      # Check for pre-existing flags set by set_source_files_properties().
      get_source_file_property(_source_file_cxx_flags ${_source_file} COMPILE_FLAGS)
      if(_source_file_cxx_flags)
        separate_arguments(_source_file_cxx_flags UNIX_COMMAND ${_source_file_cxx_flags})
        list(APPEND _source_file_cxx_flags "${_target_cxx_flags}")
      else()
        set(_source_file_cxx_flags "${_target_cxx_flags}")
      endif()
      # Apply the compile flags to the current source file.
      string(REPLACE ";" " " _source_file_cxx_flags_string "${_source_file_cxx_flags}")
      set_source_files_properties(${_source_file} PROPERTIES COMPILE_FLAGS "${_source_file_cxx_flags_string}")
    endforeach()
  endif()
  list(FIND _target_sources ${_file} _file_found_at)
  if(_file_found_at GREATER -1)
    if(NOT _cached_${_target}_${_file}_cxx_flags)
      # Cache the compile flags for the specified file.
      # This is the list that we'll be removing flags from.
      get_source_file_property(_source_file_cxx_flags ${_file} COMPILE_FLAGS)
      separate_arguments(_source_file_cxx_flags UNIX_COMMAND ${_source_file_cxx_flags})
      set(_cached_${_target}_${_file}_cxx_flags ${_source_file_cxx_flags} CACHE INTERNAL "")
    endif()
    # Remove the specified flag, then re-apply the rest.
    list(REMOVE_ITEM _cached_${_target}_${_file}_cxx_flags ${_flag})
    string(REPLACE ";" " " _cached_${_target}_${_file}_cxx_flags_string "${_cached_${_target}_${_file}_cxx_flags}")
    set_source_files_properties(${_file} PROPERTIES COMPILE_FLAGS "${_cached_${_target}_${_file}_cxx_flags_string}")
  endif()
endmacro()

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3 -mtune=native -Wno-#pragma-messages -Wno-deprecated-volatile -Wno-deprecated-declarations -Wno-deprecated-anon-enum-enum-conversion -Wno-unused-command-line-argument -Wno-#warnings -Wno-final-dtor-non-final-class")
#set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O0 -g -std=c++20 -mtune=native -Wno-#pragma-messages -Wno-deprecated-volatile -Wno-deprecated-declarations -Wno-deprecated-anon-enum-enum-conversion -Wno-unused-command-line-argument -Wno-#warnings")

add_subdirectory(libraries)
add_subdirectory(programs)
