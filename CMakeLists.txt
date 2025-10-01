cmake_minimum_required(VERSION 3.14)
project(autograd_cpp VERSION 1.0.0 LANGUAGES CXX)

# Options
option(BUILD_EXAMPLES "Build example programs" ON)
option(BUILD_TESTS "Build unit tests" ON)
option(BUILD_BENCHMARKS "Build benchmarks" ON)
option(USE_OPENMP "Use OpenMP for parallelization" ON)
option(ENABLE_AVX2 "Enable AVX2 optimizations" ON)
option(ENABLE_AVX512 "Enable AVX512 optimizations" OFF)

# C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Compiler flags
set(CMAKE_CXX_FLAGS_DEBUG "-g -O0 -DDEBUG")
set(CMAKE_CXX_FLAGS_RELEASE "-O3 -DNDEBUG")

# Architecture-specific optimizations
if(CMAKE_SYSTEM_PROCESSOR MATCHES "x86_64|AMD64")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -march=native")
    
    if(ENABLE_AVX2)
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -mavx2 -mfma")
    endif()
    
    if(ENABLE_AVX512)
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -mavx512f -mavx512dq -mavx512bw -mavx512vl")
    endif()
elseif(CMAKE_SYSTEM_PROCESSOR MATCHES "aarch64|arm64")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -mcpu=native")
endif()

# OpenMP
if(USE_OPENMP)
    find_package(OpenMP)
    if(OpenMP_CXX_FOUND)
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
    endif()
endif()

# Create interface library (header-only)
add_library(autograd INTERFACE)
target_include_directories(autograd INTERFACE 
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>
)


# Add source files if we have implementations
file(GLOB_RECURSE SOURCES "src/*.cpp")
if(SOURCES)
    add_library(autograd_impl STATIC ${SOURCES})
    target_include_directories(autograd_impl PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
        $<INSTALL_INTERFACE:include>
    )
    if(OpenMP_CXX_FOUND)
        target_link_libraries(autograd_impl PUBLIC OpenMP::OpenMP_CXX)
    endif()
endif()

# Examples
if(BUILD_EXAMPLES)
    add_subdirectory(examples)
endif()

# Tests
if(BUILD_TESTS)
    enable_testing()
    add_subdirectory(tests)
endif()

# Benchmarks
if(BUILD_BENCHMARKS)
    # add_subdirectory(benchmarks) # TODO: Add benchmarks
endif()

# Installation
include(GNUInstallDirs)

install(TARGETS autograd
    EXPORT autograd_cppTargets
)

if(TARGET autograd_impl)
    install(TARGETS autograd_impl
        EXPORT autograd_cppTargets
        ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
        LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
        RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
    )
endif()

install(DIRECTORY include/
    DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
)

# Export targets
install(EXPORT autograd_cppTargets
    FILE autograd_cppTargets.cmake
    NAMESPACE autograd::
    DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/autograd_cpp
)

# Create config file
include(CMakePackageConfigHelpers)

configure_package_config_file(
    "${CMAKE_CURRENT_SOURCE_DIR}/cmake/autograd_cppConfig.cmake.in"
    "${CMAKE_CURRENT_BINARY_DIR}/autograd_cppConfig.cmake"
    INSTALL_DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/autograd_cpp
)

write_basic_package_version_file(
    "${CMAKE_CURRENT_BINARY_DIR}/autograd_cppConfigVersion.cmake"
    VERSION ${PROJECT_VERSION}
    COMPATIBILITY SameMajorVersion
)

install(FILES
    "${CMAKE_CURRENT_BINARY_DIR}/autograd_cppConfig.cmake"
    "${CMAKE_CURRENT_BINARY_DIR}/autograd_cppConfigVersion.cmake"
    DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/autograd_cpp
)

# Print configuration
message(STATUS "")
message(STATUS "autograd-cpp configuration:")
message(STATUS "  Version:          ${PROJECT_VERSION}")
message(STATUS "  Build type:       ${CMAKE_BUILD_TYPE}")
message(STATUS "  C++ standard:     ${CMAKE_CXX_STANDARD}")
message(STATUS "  Compiler:         ${CMAKE_CXX_COMPILER_ID} ${CMAKE_CXX_COMPILER_VERSION}")
message(STATUS "  Architecture:     ${CMAKE_SYSTEM_PROCESSOR}")
message(STATUS "")
message(STATUS "  Options:")
message(STATUS "    Examples:       ${BUILD_EXAMPLES}")
message(STATUS "    Tests:          ${BUILD_TESTS}")
message(STATUS "    Benchmarks:     ${BUILD_BENCHMARKS}")
message(STATUS "    OpenMP:         ${OpenMP_CXX_FOUND}")
message(STATUS "    AVX2:           ${ENABLE_AVX2}")
message(STATUS "    AVX512:         ${ENABLE_AVX512}")
message(STATUS "")