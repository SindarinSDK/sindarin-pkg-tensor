# Overlay port for ggml — pinned to RealOrko/ggml@master with the
# repeat_back/transpose/permute backward contiguity fixes that
# sindarin-pkg-tensor's GNN training path requires.
#
# See docs/issues/ggml-issue.md for background. The patches are committed
# to RealOrko/ggml branch master at the SHA below; bump REF + SHA512
# when master is updated.

vcpkg_from_github(
    OUT_SOURCE_PATH SOURCE_PATH
    REPO RealOrko/ggml
    REF ee87371dec683446c31bd79b25b89901894400db
    SHA512 7706a23d8bb9612e2dae592c4875f773b4a10abb6ad9833d630731ac65c200a3a6839d6ea619fa43329531ee0271f5cc534957d4d70f4d26fecfa5f9b50fec4c
    HEAD_REF master
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
