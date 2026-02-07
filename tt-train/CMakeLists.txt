cmake_minimum_required(VERSION 3.18...4.2)
include(cmake/compilers.cmake)

# When building standalone (not as a tt-metal subproject) and no compiler is
# explicitly requested via CC/CXX env vars, inherit the compiler from
# tt-metal's CMakeCache.txt for ABI compatibility.
#
# Note: We cannot check CMAKE_C_COMPILER here because build systems like
# scikit-build-core pre-set it via an initial cache file (-C CMakeInit.txt)
# before CMakeLists.txt runs. The top-level check is sufficient: when tt-train
# is added as a subproject, project() has already locked in the compiler.
# When building as a standalone top-level project, try to inherit the compiler
# from tt-metal's CMakeCache.txt for ABI compatibility.
#
# We only check CMAKE_SOURCE_DIR == CMAKE_CURRENT_SOURCE_DIR (top-level check).
# We intentionally do NOT check CC/CXX env vars because build systems like
# scikit-build-core set them to generic values (e.g., "cc -pthread") that
# should be overridden with the actual compiler used to build tt-metal.
# Users who explicitly set CC/CXX to a specific compiler path can override
# via -DCMAKE_C_COMPILER on the cmake command line.
if(CMAKE_SOURCE_DIR STREQUAL CMAKE_CURRENT_SOURCE_DIR)
    inherit_compiler_from_tt_metal()
endif()

project(
    ml-framework-cpp
    LANGUAGES
        C
        CXX
)
CHECK_COMPILERS()

message(STATUS "Build type: ${CMAKE_BUILD_TYPE}")
set(CMAKE_CXX_FLAGS_RELEASE "-O3")
set(CMAKE_CXX_FLAGS_DEBUG "-O0 -g -DDEBUG=DEBUG -fno-omit-frame-pointer")
set(CMAKE_CXX_FLAGS_RELWITHDEBINFO "-O3 -g -DDEBUG=DEBUG -fno-omit-frame-pointer")
set(CMAKE_CXX_FLAGS_CI "-O3 -DDEBUG=DEBUG")

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS ON)
set(CMAKE_POSITION_INDEPENDENT_CODE TRUE)

option(ENABLE_LIBCXX "Enable using libc++" OFF)

if(CMAKE_CXX_COMPILER_ID MATCHES "Clang" AND ENABLE_LIBCXX)
    find_library(LIBC++ c++)
    find_library(LIBC++ABI c++abi)
    if(NOT LIBC++ OR NOT LIBC++ABI)
        message(
            FATAL_ERROR
            "libc++ or libc++abi not found. Make sure you have libc++ and libc++abi installed and in your PATH"
        )
    endif()
    # making it global settings for now
    add_compile_options(-stdlib=libc++)
    add_link_options(-stdlib=libc++)
endif()

message(STATUS "c++ Standard: ${CMAKE_CXX_STANDARD}")

# Add cmake directory to module path for flatbuffers helper
list(PREPEND CMAKE_MODULE_PATH ${CMAKE_CURRENT_SOURCE_DIR}/cmake)

include(cmake/dependencies.cmake)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# ttnn header files are not clean against narrowing conversion checks
# therefore tt-train likely is not clean against them
# must disable checks to build with tt-metal as super project
add_compile_options("$<$<CXX_COMPILER_ID:Clang>:-Wno-c++11-narrowing>")
add_compile_options("$<$<CXX_COMPILER_ID:GNU>:-Wno-narrowing>")

# ttml projects

add_subdirectory(sources)
include(CTest)
enable_testing()
add_subdirectory(tests)
