# Overlay port for ggml — pinned to RealOrko/ggml@sn-pkg-tensor with:
#   - repeat_back/transpose/permute backward contiguity fixes
#   - optimizer state accessor functions (m/v moments, iter counter)
#
# See docs/issues/ggml-issue.md for background. The patches are committed
# to RealOrko/ggml branch sn-pkg-tensor at the SHA below; bump REF +
# SHA512 when the branch is updated.

vcpkg_from_github(
    OUT_SOURCE_PATH SOURCE_PATH
    REPO RealOrko/ggml
    REF fb23affa3ca20528f00f8eddd25167c65ebbdb60
    SHA512 a924329fadb69e03efbd5cbc3d9d3a04318783f143ac496ee0ebe82c12c4b5f4acfd9585b716ca287f4696f2c890b36f66dff2a34cee05b2fc521aacc874fc0d
    HEAD_REF sn-pkg-tensor
)

vcpkg_check_features(OUT_FEATURE_OPTIONS FEATURE_OPTIONS
    FEATURES
        cuda  GGML_CUDA
        metal GGML_METAL
)

vcpkg_cmake_configure(
    SOURCE_PATH "${SOURCE_PATH}"
    OPTIONS
        ${FEATURE_OPTIONS}
        -DBUILD_SHARED_LIBS=OFF
        -DGGML_STATIC=ON
        # GGML_NATIVE=OFF disables -march=native and similar host-specific
        # optimization flags. Required for any distributable library build:
        # release binaries must run on consumer hardware that doesn't match
        # the build host's microarchitecture. Also fixes the macOS x86_64
        # cross-compile (the runners are now Apple Silicon M3, so
        # -march=native resolves to "apple-m3" which clang doesn't accept
        # when targeting x86_64).
        -DGGML_NATIVE=OFF
        -DGGML_BUILD_TESTS=OFF
        -DGGML_BUILD_EXAMPLES=OFF
        -DCMAKE_POSITION_INDEPENDENT_CODE=ON
)

vcpkg_cmake_install()
vcpkg_cmake_config_fixup(CONFIG_PATH lib/cmake/ggml)

# Move the headers ggml installs into a typical layout. Some ggml versions
# install headers directly under include/ rather than include/ggml/, which
# matches what sindarin-pkg-tensor's tensor.sn.c expects (#include <ggml.h>).
file(REMOVE_RECURSE
    "${CURRENT_PACKAGES_DIR}/debug/include"
    "${CURRENT_PACKAGES_DIR}/debug/share"
)

vcpkg_install_copyright(FILE_LIST "${SOURCE_PATH}/LICENSE")
