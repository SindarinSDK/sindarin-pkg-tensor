# Overlay port for ggml — pinned to RealOrko/ggml@sn-pkg-tensor with the
# repeat_back/transpose/permute backward contiguity fixes that
# sindarin-pkg-tensor's GNN training path requires.
#
# See docs/issues/ggml-issue.md for background. The patches are committed
# to RealOrko/ggml branch sn-pkg-tensor at the SHA below; bump REF + SHA512
# when the branch is updated.

vcpkg_from_github(
    OUT_SOURCE_PATH SOURCE_PATH
    REPO RealOrko/ggml
    REF 4ebc65314b17901daea2655e1525caa0efbce625
    SHA512 4ca3a5dd9a5f27c835d4cf51359a8625eaac5dc332a60bdf96d0a6dca46efd3439a972e2d16234728392815e3fb2b082d96ab0d78875c914fb19673f8a261545
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
