cmake_minimum_required(VERSION 3.20)
project(sindarin-tensor VERSION 1.0.0 LANGUAGES C)

# Allow version override from command line (e.g., -DST_VERSION=1.2.3)
if(DEFINED ST_VERSION)
    set(PROJECT_VERSION ${ST_VERSION})
endif()

message(STATUS "Sindarin Tensor libs version: ${PROJECT_VERSION}")

# Determine platform name for output directory
if(WIN32)
    set(PLATFORM_NAME "windows")
elseif(APPLE)
    set(PLATFORM_NAME "darwin")
else()
    set(PLATFORM_NAME "linux")
endif()

message(STATUS "Building for platform: ${PLATFORM_NAME}")

# Output directory structure: libs/{platform}/
set(LIBS_OUTPUT_DIR "${CMAKE_SOURCE_DIR}/libs/${PLATFORM_NAME}")
set(LIBS_LIB_DIR "${LIBS_OUTPUT_DIR}/lib")
set(LIBS_INCLUDE_DIR "${LIBS_OUTPUT_DIR}/include")

# ggml install prefix — set via -DGGML_INSTALL_PREFIX=<path>
if(NOT DEFINED GGML_INSTALL_PREFIX)
    message(FATAL_ERROR
        "GGML_INSTALL_PREFIX is not set. "
        "Run 'make build-libs' to build ggml first, then pass "
        "-DGGML_INSTALL_PREFIX=<path> to cmake.")
endif()

if(NOT EXISTS "${GGML_INSTALL_PREFIX}/lib")
    message(FATAL_ERROR "ggml libraries not found at ${GGML_INSTALL_PREFIX}/lib")
endif()

message(STATUS "ggml install prefix: ${GGML_INSTALL_PREFIX}")
message(STATUS "Output lib directory: ${LIBS_LIB_DIR}")
message(STATUS "Output include directory: ${LIBS_INCLUDE_DIR}")

# Create output directories
file(MAKE_DIRECTORY "${LIBS_LIB_DIR}")
file(MAKE_DIRECTORY "${LIBS_INCLUDE_DIR}")

# Copy all static libraries (.a files)
file(GLOB GGML_STATIC_LIBS "${GGML_INSTALL_PREFIX}/lib/*.a")
foreach(LIB ${GGML_STATIC_LIBS})
    get_filename_component(LIB_NAME ${LIB} NAME)
    message(STATUS "Copying library: ${LIB_NAME}")
    file(COPY ${LIB} DESTINATION "${LIBS_LIB_DIR}")
endforeach()

# Copy all headers
if(EXISTS "${GGML_INSTALL_PREFIX}/include")
    file(COPY "${GGML_INSTALL_PREFIX}/include/" DESTINATION "${LIBS_INCLUDE_DIR}")
    message(STATUS "Copied headers from ${GGML_INSTALL_PREFIX}/include")
endif()

# Copy pkgconfig files if they exist
if(EXISTS "${GGML_INSTALL_PREFIX}/lib/pkgconfig")
    file(MAKE_DIRECTORY "${LIBS_OUTPUT_DIR}/lib/pkgconfig")
    file(COPY "${GGML_INSTALL_PREFIX}/lib/pkgconfig/" DESTINATION "${LIBS_OUTPUT_DIR}/lib/pkgconfig")
    message(STATUS "Copied pkgconfig files")
endif()

# Create metadata files
file(WRITE "${LIBS_OUTPUT_DIR}/VERSION" "${PROJECT_VERSION}\n")
file(WRITE "${LIBS_OUTPUT_DIR}/PLATFORM" "${PLATFORM_NAME}\n")

# Create a manifest of included libraries
file(GLOB ALL_LIBS "${LIBS_LIB_DIR}/*.a")
set(LIB_MANIFEST "")
foreach(LIB ${ALL_LIBS})
    get_filename_component(LIB_NAME ${LIB} NAME)
    string(APPEND LIB_MANIFEST "${LIB_NAME}\n")
endforeach()
file(WRITE "${LIBS_OUTPUT_DIR}/LIBRARIES.txt" "${LIB_MANIFEST}")

message(STATUS "")
message(STATUS "========================================")
message(STATUS "Sindarin Tensor libs build complete!")
message(STATUS "Platform: ${PLATFORM_NAME}")
message(STATUS "Output directory: ${LIBS_OUTPUT_DIR}")
message(STATUS "========================================")

add_custom_target(install-libs
    COMMAND ${CMAKE_COMMAND} -E echo "Libraries installed to ${LIBS_OUTPUT_DIR}"
)
