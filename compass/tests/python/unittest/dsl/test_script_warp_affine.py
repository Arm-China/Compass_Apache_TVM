# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2023-2025 Arm Technology (China) Co. Ltd.
import cv2
import numpy as np
from tvm.compass.dsl import BuildManager, script as S
from tvm.compass.dsl.testing import assert_allclose
from tvm.compass.relax.testing import get_imagenet_input


@S.prim_func
def warp_affine(
    img_ptr: S.ptr("u8", "global"),
    mat_ptr: S.ptr("fp32", "global"),
    out_ptr: S.ptr("u8", "global"),
):
    img = S.match_buffer(img_ptr, (256, 256, 3))
    mat = S.match_buffer(mat_ptr, (2, 3))
    out = S.match_buffer(out_ptr, (256, 256, 3))

    for ty in S.tec_range(4):
        for yi in range(256 // 4):
            for x in range(256):
                y = ty * (256 // 4) + yi
                u = S.i32(x * mat[0, 0] + y * mat[0, 1] + mat[0, 2])
                v = S.i32(x * mat[1, 0] + y * mat[1, 1] + mat[1, 2])

                if u < 0 or u > 255 or v < 0 or v > 255:
                    out[y, x, 0] = 0
                    out[y, x, 1] = 0
                    out[y, x, 2] = 0
                else:
                    out[y, x, 0] = img[v, u, 0]
                    out[y, x, 1] = img[v, u, 1]
                    out[y, x, 2] = img[v, u, 2]


def test_warp_affine():
    img = get_imagenet_input(256, 256, None).astype(np.uint8)
    img = img.reshape((256, 256, 3))
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

    x_scale = 2
    y_scale = 1.5
    theta = np.pi / 3
    tx = 150
    ty = -50

    mat_fp = np.array(
        [
            [np.cos(theta) * x_scale, -np.sin(theta) * y_scale, tx],
            [np.sin(theta) * x_scale, np.cos(theta) * y_scale, ty],
        ]
    ).astype(np.float32)

    bm = BuildManager()
    ex = bm.build(warp_affine)

    py_out = np.empty((256, 256, 3), dtype=np.uint8)
    warp_affine(img, mat_fp, py_out)
    # cv2.imwrite("cat_rotate_gt.jpg", py_out)

    npu_out = np.empty((256, 256, 3), dtype=np.uint8)
    ex(img, mat_fp, npu_out)
    # cv2.imwrite("cat_rotate_npu.jpg", npu_out)
    assert_allclose(npu_out, py_out)


if __name__ == "__main__":
    test_warp_affine()
