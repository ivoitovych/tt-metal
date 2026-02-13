# Compiler selection and validation for tt-train.
#
# When building standalone (not as a tt-metal subproject), tt-train needs
# a compiler compatible with the one used to build tt-metal, since it links
# against tt-metal libraries (libtt_metal.so, _ttnncpp.so).
#
# Compiler selection priority (before project()):
#   1. -DCMAKE_C_COMPILER=... on the cmake command line (or toolchain file)
#   2. CC/CXX environment variables
#   3. Inherited from tt-metal's CMakeCache.txt
#   4. CMake default search (CHECK_COMPILERS() validates the result)
#
# Restored after deletion in a7618b9282 ("TT-Train: bump clang version
# from 17 to 20 #36568").
#
# See: https://github.com/tenstorrent/tt-metal/issues/36993

# Read the compiler used by tt-metal from its CMakeCache.txt and set it
# as CMAKE_C_COMPILER/CMAKE_CXX_COMPILER so tt-train uses the same one.
function(INHERIT_COMPILER_FROM_TT_METAL)
    # Determine TT_METAL_HOME
    if(DEFINED ENV{TT_METAL_HOME})
        set(_tt_metal_home "$ENV{TT_METAL_HOME}")
    else()
        # Infer from directory structure: tt-metal/tt-train/ -> tt-metal/
        set(_tt_metal_home "${CMAKE_CURRENT_SOURCE_DIR}/..")
        if(NOT EXISTS "${_tt_metal_home}/tt_metal/CMakeLists.txt")
            message(STATUS "Cannot infer tt-metal location; using CMake default compiler")
            return()
        endif()
    endif()

    # Search for CMakeCache.txt in known build directories
    set(_cache_file "")
    foreach(_build_dir "build" "build_Debug" "build_Release")
        if(EXISTS "${_tt_metal_home}/${_build_dir}/CMakeCache.txt")
            set(_cache_file "${_tt_metal_home}/${_build_dir}/CMakeCache.txt")
            break()
        endif()
    endforeach()

    if(NOT _cache_file)
        message(STATUS "No tt-metal build found; using CMake default compiler")
        return()
    endif()

    # Parse CMAKE_C_COMPILER and CMAKE_CXX_COMPILER from CMakeCache.txt
    file(STRINGS "${_cache_file}" _c_compiler_line REGEX "^CMAKE_C_COMPILER:.*=")
    file(STRINGS "${_cache_file}" _cxx_compiler_line REGEX "^CMAKE_CXX_COMPILER:.*=")

    if(_c_compiler_line AND _cxx_compiler_line)
        string(REGEX REPLACE "^CMAKE_C_COMPILER:[^=]*=(.*)" "\\1" _c_compiler "${_c_compiler_line}")
        string(REGEX REPLACE "^CMAKE_CXX_COMPILER:[^=]*=(.*)" "\\1" _cxx_compiler "${_cxx_compiler_line}")

        if(EXISTS "${_c_compiler}" AND EXISTS "${_cxx_compiler}")
            message(STATUS "Inheriting compiler from tt-metal build (${_cache_file}):")
            message(STATUS "  C compiler:   ${_c_compiler}")
            message(STATUS "  C++ compiler: ${_cxx_compiler}")
            set(CMAKE_C_COMPILER "${_c_compiler}" PARENT_SCOPE)
            set(CMAKE_CXX_COMPILER "${_cxx_compiler}" PARENT_SCOPE)
        else()
            message(STATUS "Compiler paths from tt-metal CMakeCache not found on disk; using CMake default")
        endif()
    else()
        message(STATUS "Could not parse compiler from ${_cache_file}; using CMake default")
    endif()
endfunction()

function(CHECK_COMPILERS)
    message(STATUS "Checking compilers: ${CMAKE_CXX_COMPILER_ID} ${CMAKE_CXX_COMPILER_VERSION}")

    if(CMAKE_CXX_COMPILER_ID STREQUAL "Clang")
        if(CMAKE_CXX_COMPILER_VERSION VERSION_LESS "17.0.0")
            message(WARNING "Clang-17 or higher is recommended")
        endif()
    elseif(CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
        if(CMAKE_CXX_COMPILER_VERSION VERSION_LESS "12.0.0")
            message(FATAL_ERROR "GCC-12 or higher is required")
        elseif(CMAKE_CXX_COMPILER_VERSION GREATER_EQUAL "13.0.0")
            message(WARNING "Only GCC-12 is tested right now")
        endif()
    else()
        message(FATAL_ERROR "Unsupported compiler: ${CMAKE_CXX_COMPILER_ID} ! Only Clang and GCC are supported")
    endif()
endfunction()

function(ADJUST_COMPILER_WARNINGS)
    if(CMAKE_CXX_COMPILER_ID STREQUAL "Clang")
        target_compile_options(
            compiler_warnings
            INTERFACE
                -Wsometimes-uninitialized
                -Wno-c++11-narrowing
                -Wno-error=local-type-template-args
                -Wno-delete-non-abstract-non-virtual-dtor
                -Wno-c99-designator
                -Wno-shift-op-parentheses
                -Wno-non-c-typedef-for-linkage
                -Wno-deprecated-this-capture
                -Wno-deprecated-volatile
                -Wno-deprecated-builtins
                -Wno-deprecated-declarations
        )
    else() # GCC-12 or higher
        target_compile_options(
            compiler_warnings
            INTERFACE
                -Wno-deprecated
                -Wno-attributes
                -Wno-stringop-overread
                -Wno-stringop-overflow
                -Wno-maybe-uninitialized
                -Wno-missing-requires
                -Wno-narrowing
                -Wno-non-template-friend
                -Wno-error=non-template-friend
        )
    endif()
endfunction()
