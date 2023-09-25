# This file is CONFIDENTIAL and created by Arm Technology (China) Co., Ltd.
# See the copyright file distributed with this work for additional information
# regarding copyright ownership.
import numpy as np
import tvm
from tvm import relay
from tvm.relay.backend.contrib.aipu_compass import transform as compass_transform
import tvm.testing

ADT_module = """
#[version = "0.0.5"]
type List[A] {
  Cons(A, List[A]),
  Nil,
}

type Option[A] {
  Some(A),
  None,
}

type Tree[A] {
  Rose(A, List[Tree[A]]),
}

type static_tensor_float32_100_4_t {
  tensor_nil_float32_100_4,
  tensor_constructor_float32_100_4(Tensor[(100, 4), float32]),
}

type static_tensor_float32_100_90_4_t {
  tensor_nil_float32_100_90_4,
  tensor_constructor_float32_100_90_4(Tensor[(100, 90, 4), float32]),
}

type static_tensor_float32_100_90_t {
  tensor_nil_float32_100_90,
  tensor_constructor_float32_100_90(Tensor[(100, 90), float32]),
}

type static_tensor_float32_100_t {
  tensor_nil_float32_100,
  tensor_constructor_float32_100(Tensor[(100), float32]),
}

type static_tensor_float32_17328_1_4_t {
  tensor_nil_float32_17328_1_4,
  tensor_constructor_float32_17328_1_4(Tensor[(17328, 1, 4), float32]),
}

type static_tensor_float32_17328_1_t {
  tensor_nil_float32_17328_1,
  tensor_constructor_float32_17328_1(Tensor[(17328, 1), float32]),
}

type static_tensor_float32_1_100_4_t {
  tensor_nil_float32_1_100_4,
  tensor_constructor_float32_1_100_4(Tensor[(1, 100, 4), float32]),
}

type static_tensor_float32_1_100_90_4_t {
  tensor_nil_float32_1_100_90_4,
  tensor_constructor_float32_1_100_90_4(Tensor[(1, 100, 90, 4), float32]),
}

type static_tensor_float32_1_100_90_t {
  tensor_nil_float32_1_100_90,
  tensor_constructor_float32_1_100_90(Tensor[(1, 100, 90), float32]),
}

type static_tensor_float32_1_17328_1_4_t {
  tensor_nil_float32_1_17328_1_4,
  tensor_constructor_float32_1_17328_1_4(Tensor[(1, 17328, 1, 4), float32]),
}

type static_tensor_float32_1_17328_1_t {
  tensor_nil_float32_1_17328_1,
  tensor_constructor_float32_1_17328_1(Tensor[(1, 17328, 1), float32]),
}

type static_tensor_float32_1_4_t {
  tensor_nil_float32_1_4,
  tensor_constructor_float32_1_4(Tensor[(1, 4), float32]),
}

type static_tensor_float32_1_512_512_3_t {
  tensor_nil_float32_1_512_512_3,
  tensor_constructor_float32_1_512_512_3(Tensor[(1, 512, 512, 3), float32]),
}

type static_tensor_float32_1_t {
  tensor_nil_float32_1,
  tensor_constructor_float32_1(Tensor[(1), float32]),
}

type static_tensor_float32_4_t {
  tensor_nil_float32_4,
  tensor_constructor_float32_4(Tensor[(4), float32]),
}

type static_tensor_float32_512_3_t {
  tensor_nil_float32_512_3,
  tensor_constructor_float32_512_3(Tensor[(512, 3), float32]),
}

type static_tensor_float32_512_512_3_t {
  tensor_nil_float32_512_512_3,
  tensor_constructor_float32_512_512_3(Tensor[(512, 512, 3), float32]),
}

type static_tensor_float32_600_3_t {
  tensor_nil_float32_600_3,
  tensor_constructor_float32_600_3(Tensor[(600, 3), float32]),
}

type static_tensor_float32_600_600_3_t {
  tensor_nil_float32_600_600_3,
  tensor_constructor_float32_600_600_3(Tensor[(600, 600, 3), float32]),
}

type static_tensor_float32_90_4_t {
  tensor_nil_float32_90_4,
  tensor_constructor_float32_90_4(Tensor[(90, 4), float32]),
}

type static_tensor_float32_90_t {
  tensor_nil_float32_90,
  tensor_constructor_float32_90(Tensor[(90), float32]),
}

type static_tensor_float32_any_100_4_t {
  tensor_nil_float32_any_100_4,
  tensor_constructor_float32_any_100_4(Tensor[(?, 100, 4), float32]),
}

type static_tensor_float32_any_100_90_4_t {
  tensor_nil_float32_any_100_90_4,
  tensor_constructor_float32_any_100_90_4(Tensor[(?, 100, 90, 4), float32]),
}

type static_tensor_float32_any_100_90_t {
  tensor_nil_float32_any_100_90,
  tensor_constructor_float32_any_100_90(Tensor[(?, 100, 90), float32]),
}

type static_tensor_float32_any_100_t {
  tensor_nil_float32_any_100,
  tensor_constructor_float32_any_100(Tensor[(?, 100), float32]),
}

type static_tensor_float32_any_17328_1_4_t {
  tensor_nil_float32_any_17328_1_4,
  tensor_constructor_float32_any_17328_1_4(Tensor[(?, 17328, 1, 4), float32]),
}

type static_tensor_float32_any_17328_1_t {
  tensor_nil_float32_any_17328_1,
  tensor_constructor_float32_any_17328_1(Tensor[(?, 17328, 1), float32]),
}

type static_tensor_float32_any_1_100_4_t {
  tensor_nil_float32_any_1_100_4,
  tensor_constructor_float32_any_1_100_4(Tensor[(?, 1, 100, 4), float32]),
}

type static_tensor_float32_any_1_100_90_4_t {
  tensor_nil_float32_any_1_100_90_4,
  tensor_constructor_float32_any_1_100_90_4(Tensor[(?, 1, 100, 90, 4), float32]),
}

type static_tensor_float32_any_1_100_90_t {
  tensor_nil_float32_any_1_100_90,
  tensor_constructor_float32_any_1_100_90(Tensor[(?, 1, 100, 90), float32]),
}

type static_tensor_float32_any_1_17328_1_4_t {
  tensor_nil_float32_any_1_17328_1_4,
  tensor_constructor_float32_any_1_17328_1_4(Tensor[(?, 1, 17328, 1, 4), float32]),
}

type static_tensor_float32_any_1_17328_1_t {
  tensor_nil_float32_any_1_17328_1,
  tensor_constructor_float32_any_1_17328_1(Tensor[(?, 1, 17328, 1), float32]),
}

type static_tensor_float32_any_1_4_t {
  tensor_nil_float32_any_1_4,
  tensor_constructor_float32_any_1_4(Tensor[(?, 1, 4), float32]),
}

type static_tensor_float32_any_1_512_512_3_t {
  tensor_nil_float32_any_1_512_512_3,
  tensor_constructor_float32_any_1_512_512_3(Tensor[(?, 1, 512, 512, 3), float32]),
}

type static_tensor_float32_any_1_t {
  tensor_nil_float32_any_1,
  tensor_constructor_float32_any_1(Tensor[(?, 1), float32]),
}

type static_tensor_float32_any_4_t {
  tensor_nil_float32_any_4,
  tensor_constructor_float32_any_4(Tensor[(?, 4), float32]),
}

type static_tensor_float32_any_512_3_t {
  tensor_nil_float32_any_512_3,
  tensor_constructor_float32_any_512_3(Tensor[(?, 512, 3), float32]),
}

type static_tensor_float32_any_512_512_3_t {
  tensor_nil_float32_any_512_512_3,
  tensor_constructor_float32_any_512_512_3(Tensor[(?, 512, 512, 3), float32]),
}

type static_tensor_float32_any_600_3_t {
  tensor_nil_float32_any_600_3,
  tensor_constructor_float32_any_600_3(Tensor[(?, 600, 3), float32]),
}

type static_tensor_float32_any_600_600_3_t {
  tensor_nil_float32_any_600_600_3,
  tensor_constructor_float32_any_600_600_3(Tensor[(?, 600, 600, 3), float32]),
}

type static_tensor_float32_any_90_4_t {
  tensor_nil_float32_any_90_4,
  tensor_constructor_float32_any_90_4(Tensor[(?, 90, 4), float32]),
}

type static_tensor_float32_any_90_t {
  tensor_nil_float32_any_90,
  tensor_constructor_float32_any_90(Tensor[(?, 90), float32]),
}

type static_tensor_float32_any_t {
  tensor_nil_float32_any,
  tensor_constructor_float32_any(Tensor[(?), float32]),
}

type static_tensor_float32_scalar_t {
  tensor_nil_float32_scalar,
  tensor_constructor_float32_scalar(float32),
}

type static_tensor_int32_1_3_t {
  tensor_nil_int32_1_3,
  tensor_constructor_int32_1_3(Tensor[(1, 3), int32]),
}

type static_tensor_int32_1_t {
  tensor_nil_int32_1,
  tensor_constructor_int32_1(Tensor[(1), int32]),
}

type static_tensor_int32_3_t {
  tensor_nil_int32_3,
  tensor_constructor_int32_3(Tensor[(3), int32]),
}

type static_tensor_int32_any_1_3_t {
  tensor_nil_int32_any_1_3,
  tensor_constructor_int32_any_1_3(Tensor[(?, 1, 3), int32]),
}

type static_tensor_int32_any_1_t {
  tensor_nil_int32_any_1,
  tensor_constructor_int32_any_1(Tensor[(?, 1), int32]),
}

type static_tensor_int32_any_3_t {
  tensor_nil_int32_any_3,
  tensor_constructor_int32_any_3(Tensor[(?, 3), int32]),
}

type static_tensor_int32_any_t {
  tensor_nil_int32_any,
  tensor_constructor_int32_any(Tensor[(?), int32]),
}

type static_tensor_int32_scalar_t {
  tensor_nil_int32_scalar,
  tensor_constructor_int32_scalar(int32),
}

type tensor_float16_t {
  tensor_nil_float16,
  tensor0_float16(float16),
  tensor1_float16(Tensor[(?), float16]),
  tensor2_float16(Tensor[(?, ?), float16]),
  tensor3_float16(Tensor[(?, ?, ?), float16]),
  tensor4_float16(Tensor[(?, ?, ?, ?), float16]),
  tensor5_float16(Tensor[(?, ?, ?, ?, ?), float16]),
  tensor6_float16(Tensor[(?, ?, ?, ?, ?, ?), float16]),
}

type tensor_float32_t {
  tensor_nil_float32,
  tensor0_float32(float32),
  tensor1_float32(Tensor[(?), float32]),
  tensor2_float32(Tensor[(?, ?), float32]),
  tensor3_float32(Tensor[(?, ?, ?), float32]),
  tensor4_float32(Tensor[(?, ?, ?, ?), float32]),
  tensor5_float32(Tensor[(?, ?, ?, ?, ?), float32]),
  tensor6_float32(Tensor[(?, ?, ?, ?, ?, ?), float32]),
}

type tensor_float64_t {
  tensor_nil_float64,
  tensor0_float64(float64),
  tensor1_float64(Tensor[(?), float64]),
  tensor2_float64(Tensor[(?, ?), float64]),
  tensor3_float64(Tensor[(?, ?, ?), float64]),
  tensor4_float64(Tensor[(?, ?, ?, ?), float64]),
  tensor5_float64(Tensor[(?, ?, ?, ?, ?), float64]),
  tensor6_float64(Tensor[(?, ?, ?, ?, ?, ?), float64]),
}

type tensor_int16_t {
  tensor_nil_int16,
  tensor0_int16(int16),
  tensor1_int16(Tensor[(?), int16]),
  tensor2_int16(Tensor[(?, ?), int16]),
  tensor3_int16(Tensor[(?, ?, ?), int16]),
  tensor4_int16(Tensor[(?, ?, ?, ?), int16]),
  tensor5_int16(Tensor[(?, ?, ?, ?, ?), int16]),
  tensor6_int16(Tensor[(?, ?, ?, ?, ?, ?), int16]),
}

type tensor_int32_t {
  tensor_nil_int32,
  tensor0_int32(int32),
  tensor1_int32(Tensor[(?), int32]),
  tensor2_int32(Tensor[(?, ?), int32]),
  tensor3_int32(Tensor[(?, ?, ?), int32]),
  tensor4_int32(Tensor[(?, ?, ?, ?), int32]),
  tensor5_int32(Tensor[(?, ?, ?, ?, ?), int32]),
  tensor6_int32(Tensor[(?, ?, ?, ?, ?, ?), int32]),
}

type tensor_int64_t {
  tensor_nil_int64,
  tensor0_int64(int64),
  tensor1_int64(Tensor[(?), int64]),
  tensor2_int64(Tensor[(?, ?), int64]),
  tensor3_int64(Tensor[(?, ?, ?), int64]),
  tensor4_int64(Tensor[(?, ?, ?, ?), int64]),
  tensor5_int64(Tensor[(?, ?, ?, ?, ?), int64]),
  tensor6_int64(Tensor[(?, ?, ?, ?, ?, ?), int64]),
}

type tensor_int8_t {
  tensor_nil_int8,
  tensor0_int8(int8),
  tensor1_int8(Tensor[(?), int8]),
  tensor2_int8(Tensor[(?, ?), int8]),
  tensor3_int8(Tensor[(?, ?, ?), int8]),
  tensor4_int8(Tensor[(?, ?, ?, ?), int8]),
  tensor5_int8(Tensor[(?, ?, ?, ?, ?), int8]),
  tensor6_int8(Tensor[(?, ?, ?, ?, ?, ?), int8]),
}

type tensor_uint16_t {
  tensor_nil_uint16,
  tensor0_uint16(uint16),
  tensor1_uint16(Tensor[(?), uint16]),
  tensor2_uint16(Tensor[(?, ?), uint16]),
  tensor3_uint16(Tensor[(?, ?, ?), uint16]),
  tensor4_uint16(Tensor[(?, ?, ?, ?), uint16]),
  tensor5_uint16(Tensor[(?, ?, ?, ?, ?), uint16]),
  tensor6_uint16(Tensor[(?, ?, ?, ?, ?, ?), uint16]),
}

type tensor_uint8_t {
  tensor_nil_uint8,
  tensor0_uint8(uint8),
  tensor1_uint8(Tensor[(?), uint8]),
  tensor2_uint8(Tensor[(?, ?), uint8]),
  tensor3_uint8(Tensor[(?, ?, ?), uint8]),
  tensor4_uint8(Tensor[(?, ?, ?, ?), uint8]),
  tensor5_uint8(Tensor[(?, ?, ?, ?, ?), uint8]),
  tensor6_uint8(Tensor[(?, ?, ?, ?, ?, ?), uint8]),
}

def @hd[A](%xs5: List[A] /* ty=List[A] span=from_string:376:11 */) -> A {
  match? (%xs5) {
    Cons(%x6: A /* ty=A span=from_string:377:5 */, _) => {
      %x6
    },
  }
}

def @main(%image_tensor: Tensor[(1, 512, 512, 3), uint8] /* ty=Tensor[(1, 512, 512, 3), uint8] span=from_string:1462:16 */) {
  %0 = cast(%image_tensor, dtype="float32") /* ty=Tensor[(1, 512, 512, 3), float32] span=from_string:413:51 */;
  %1 = @tensor_array_float32_512_512_3(1 /* ty=int32 span=from_string:412:42 */) /* ty=List[static_tensor_float32_512_512_3_t[]] span=from_string:414:49 */;
  %2 = @tensor_array_unstack_float32_1_512_512_3(%0) /* ty=List[static_tensor_float32_512_512_3_t[]] span=from_string:414:133 */;
  %3 = @tensor_array_scatter_float32_512_512_3(%1, meta[relay.Constant][0] /* ty=Tensor[(1), int32] span=from_string:1465:62 */, %2) /* ty=List[static_tensor_float32_512_512_3_t[]] span=from_string:415:46 */;
  %4 = @tensor_array_read_float32_512_512_3(%3, 0 /* ty=int32 span=from_string:415:52 */) /* ty=static_tensor_float32_512_512_3_t[] span=from_string:416:44 */;
  %5 = @tensor_get_data_float32_512_512_3(%4) /* ty=Tensor[(512, 512, 3), float32] span=from_string:417:21 */;
  %6 = expand_dims(%5, axis=0) /* ty=Tensor[(1, 512, 512, 3), float32] span=from_string:418:25 */;
  %7 = image.resize2d(%6, size=[600, 600], roi=[0f, 0f, 0f, 0f], layout="NHWC", coordinate_transformation_mode="asymmetric", rounding_method="") /* ty=Tensor[(1, 600, 600, 3), float32] span=from_string:419:17 */;
  %8 = squeeze(%7, axis=[0]) /* ty=Tensor[(600, 600, 3), float32] span=from_string:421:46 */;
  %9 = @tensor_array_float32_600_600_3(1 /* ty=int32 span=from_string:420:42 */) /* ty=List[static_tensor_float32_600_600_3_t[]] span=from_string:422:47 */;
  %10 = tensor_constructor_float32_600_600_3(%8) /* ty=static_tensor_float32_600_600_3_t[] span=from_string:422:70 */;
  %11 = @tensor_array_write_float32_600_600_3(%9, 0 /* ty=int32 span=from_string:422:53 */, %10) /* ty=List[static_tensor_float32_600_600_3_t[]] span=from_string:423:46 */;
  %12 = @tensor_array_read_float32_600_600_3(%11, 0 /* ty=int32 span=from_string:423:52 */) /* ty=static_tensor_float32_600_600_3_t[] span=from_string:424:44 */;
  @tensor_get_data_float32_600_600_3(%12) /* ty=Tensor[(600, 600, 3), float32] span=from_string:425:21 */
}

def @nth[A](%xs10: List[A] /* ty=List[A] span=from_string:10136:17 */, %n1: int32 /* ty=int32 span=from_string:10137:22 */) -> A {
  %13 = equal(%n1, 0 /* ty=int32 span=from_string:10132:23 */) /* ty=bool span=from_string:10133:7 */;
  if (%13) {
    @hd(%xs10) /* ty=A span=from_string:10134:5 */
  } else {
    %14 = @tl(%xs10) /* ty=List[A] span=from_string:10138:10 */;
    %15 = subtract(%n1, 1 /* ty=int32 span=from_string:10137:28 */) /* ty=int32 span=from_string:10138:17 */;
    @nth(%14, %15) /* ty=A span=from_string:10136:5 */
  }
}

def @tensor_array_float32_512_512_3(%x28: int32 /* ty=int32 span=from_string:10231:22 */) -> List[static_tensor_float32_512_512_3_t[]] {
  %16 = equal(%x28, 0 /* ty=int32 span=from_string:10227:24 */) /* ty=bool span=from_string:10228:7 */;
  if (%16) {
    Nil /* ty=List[static_tensor_float32_512_512_3_t[]] span=from_string:10229:5 */
  } else {
    %17 = subtract(%x28, 1 /* ty=int32 span=from_string:10231:29 */) /* ty=int32 span=from_string:10233:45 */;
    %18 = tensor_nil_float32_512_512_3 /* ty=static_tensor_float32_512_512_3_t[] span=from_string:10234:10 */;
    %19 = @tensor_array_float32_512_512_3(%17) /* ty=List[static_tensor_float32_512_512_3_t[]] span=from_string:10234:17 */;
    Cons(%18, %19) /* ty=List[static_tensor_float32_512_512_3_t[]] span=from_string:10231:5 */
  }
}

def @tensor_array_float32_600_600_3(%x29: int32 /* ty=int32 span=from_string:10243:22 */) -> List[static_tensor_float32_600_600_3_t[]] {
  %20 = equal(%x29, 0 /* ty=int32 span=from_string:10239:24 */) /* ty=bool span=from_string:10240:7 */;
  if (%20) {
    Nil /* ty=List[static_tensor_float32_600_600_3_t[]] span=from_string:10241:5 */
  } else {
    %21 = subtract(%x29, 1 /* ty=int32 span=from_string:10243:29 */) /* ty=int32 span=from_string:10245:45 */;
    %22 = tensor_nil_float32_600_600_3 /* ty=static_tensor_float32_600_600_3_t[] span=from_string:10246:10 */;
    %23 = @tensor_array_float32_600_600_3(%21) /* ty=List[static_tensor_float32_600_600_3_t[]] span=from_string:10246:17 */;
    Cons(%22, %23) /* ty=List[static_tensor_float32_600_600_3_t[]] span=from_string:10243:5 */
  }
}

def @tensor_array_read_float32_512_512_3(%tensor_array92: List[static_tensor_float32_512_512_3_t[]] /* ty=List[static_tensor_float32_512_512_3_t[]] span=from_string:10303:8 */, %x55: int32 /* ty=int32 span=from_string:10303:25 */) -> static_tensor_float32_512_512_3_t[] {
  @nth(%tensor_array92, %x55) /* ty=static_tensor_float32_512_512_3_t[] span=from_string:10303:3 */
}

def @tensor_array_read_float32_600_600_3(%tensor_array93: List[static_tensor_float32_600_600_3_t[]] /* ty=List[static_tensor_float32_600_600_3_t[]] span=from_string:10307:8 */, %x56: int32 /* ty=int32 span=from_string:10307:25 */) -> static_tensor_float32_600_600_3_t[] {
  @nth(%tensor_array93, %x56) /* ty=static_tensor_float32_600_600_3_t[] span=from_string:10307:3 */
}

def @tensor_array_scatter_float32_512_512_3(%tensor_array127: List[static_tensor_float32_512_512_3_t[]] /* ty=List[static_tensor_float32_512_512_3_t[]] span=from_string:10357:50 */, %indices42: Tensor[(?), int32] /* ty=Tensor[(?), int32] span=from_string:10357:119 */, %values22: List[static_tensor_float32_512_512_3_t[]] /* ty=List[static_tensor_float32_512_512_3_t[]] span=from_string:10357:133 */) -> List[static_tensor_float32_512_512_3_t[]] {
  %24 = shape_of(%indices42, dtype="int32") /* ty=Tensor[(1), int32] span=from_string:10356:16 */;
  %25 = take(%24, 0 /* ty=int32 span=from_string:10356:24 */) /* ty=int32 span=from_string:10357:112 */;
  @tensor_array_scatter_helper_float32_512_512_3(%tensor_array127, 0 /* ty=int32 span=from_string:10357:69 */, %25, %indices42, %values22) /* ty=List[static_tensor_float32_512_512_3_t[]] span=from_string:10355:3 */
}

def @tensor_array_scatter_helper_float32_512_512_3(%ta42: List[static_tensor_float32_512_512_3_t[]] /* ty=List[static_tensor_float32_512_512_3_t[]] span=from_string:10445:51 */, %current52: int32 /* ty=int32 span=from_string:10446:17 */, %limit52: int32 /* ty=int32 span=from_string:10447:66 */, %indices_42: Tensor[(?), int32] /* ty=Tensor[(?), int32] span=from_string:10447:76 */, %values_22: List[static_tensor_float32_512_512_3_t[]] /* ty=List[static_tensor_float32_512_512_3_t[]] span=from_string:10447:91 */) -> List[static_tensor_float32_512_512_3_t[]] {
  %26 = equal(%current52, %limit52) /* ty=bool span=from_string:10440:7 */;
  if (%26) {
    %ta42
  } else {
    %27 = take(%indices_42, %current52) /* ty=int32 span=from_string:10445:58 */;
    %28 = @tensor_array_read_float32_512_512_3(%values_22, %current52) /* ty=static_tensor_float32_512_512_3_t[] span=from_string:10445:65 */;
    %29 = @tensor_array_write_float32_512_512_3(%ta42, %27, %28) /* ty=List[static_tensor_float32_512_512_3_t[]] span=from_string:10447:52 */;
    %30 = add(%current52, 1 /* ty=int32 span=from_string:10446:30 */) /* ty=int32 span=from_string:10447:59 */;
    @tensor_array_scatter_helper_float32_512_512_3(%29, %30, %limit52, %indices_42, %values_22) /* ty=List[static_tensor_float32_512_512_3_t[]] span=from_string:10443:5 */
  }
}

def @tensor_array_unstack_float32_1_512_512_3(%tensor12: Tensor[(1, 512, 512, 3), float32] /* ty=Tensor[(1, 512, 512, 3), float32] span=from_string:10514:114 */) -> List[static_tensor_float32_512_512_3_t[]] {
  @tensor_array_unstack_helper_float32_1_512_512_3(0 /* ty=int32 span=from_string:10514:53 */, 1 /* ty=int32 span=from_string:10514:97 */, %tensor12) /* ty=List[static_tensor_float32_512_512_3_t[]] span=from_string:10514:3 */
}

def @tensor_array_unstack_helper_float32_1_512_512_3(%i12: int32 /* ty=int32 span=from_string:10601:17 */, %up12: int32 /* ty=int32 span=from_string:10603:69 */, %t14: Tensor[(1, 512, 512, 3), float32] /* ty=Tensor[(1, 512, 512, 3), float32] span=from_string:10603:76 */) -> List[static_tensor_float32_512_512_3_t[]] {
  %31 = equal(%i12, %up12) /* ty=bool span=from_string:10597:7 */;
  if (%31) {
    Nil /* ty=List[static_tensor_float32_512_512_3_t[]] span=from_string:10598:5 */
  } else {
    %32 = take(%t14, %i12, axis=0) /* ty=Tensor[(512, 512, 3), float32] span=from_string:10602:50 */;
    %33 = add(%i12, 1 /* ty=int32 span=from_string:10601:25 */) /* ty=int32 span=from_string:10603:62 */;
    %34 = tensor_constructor_float32_512_512_3(%32) /* ty=static_tensor_float32_512_512_3_t[] span=from_string:10604:10 */;
    %35 = @tensor_array_unstack_helper_float32_1_512_512_3(%33, %up12, %t14) /* ty=List[static_tensor_float32_512_512_3_t[]] span=from_string:10604:17 */;
    Cons(%34, %35) /* ty=List[static_tensor_float32_512_512_3_t[]] span=from_string:10600:5 */
  }
}

def @tensor_array_write_float32_512_512_3(%tensor_array216: List[static_tensor_float32_512_512_3_t[]] /* ty=List[static_tensor_float32_512_512_3_t[]] span=from_string:10671:11 */, %x86: int32 /* ty=int32 span=from_string:10671:29 */, %v16: static_tensor_float32_512_512_3_t[] /* ty=static_tensor_float32_512_512_3_t[] span=from_string:10671:35 */) -> List[static_tensor_float32_512_512_3_t[]] {
  @update(%tensor_array216, %x86, %v16) /* ty=List[static_tensor_float32_512_512_3_t[]] span=from_string:10671:3 */
}

def @tensor_array_write_float32_600_600_3(%tensor_array217: List[static_tensor_float32_600_600_3_t[]] /* ty=List[static_tensor_float32_600_600_3_t[]] span=from_string:10675:11 */, %x87: int32 /* ty=int32 span=from_string:10675:29 */, %v17: static_tensor_float32_600_600_3_t[] /* ty=static_tensor_float32_600_600_3_t[] span=from_string:10675:35 */) -> List[static_tensor_float32_600_600_3_t[]] {
  @update(%tensor_array217, %x87, %v17) /* ty=List[static_tensor_float32_600_600_3_t[]] span=from_string:10675:3 */
}

def @tensor_get_data_float32_512_512_3(%tensor87: static_tensor_float32_512_512_3_t[] /* ty=static_tensor_float32_512_512_3_t[] span=from_string:10743:11 */) -> Tensor[(512, 512, 3), float32] {
  match? (%tensor87) {
    tensor_constructor_float32_512_512_3(%t109: Tensor[(512, 512, 3), float32] /* ty=Tensor[(512, 512, 3), float32] span=from_string:10744:5 */) => {
      %t109
    },
  }
}

def @tensor_get_data_float32_600_600_3(%tensor88: static_tensor_float32_600_600_3_t[] /* ty=static_tensor_float32_600_600_3_t[] span=from_string:10751:11 */) -> Tensor[(600, 600, 3), float32] {
  match? (%tensor88) {
    tensor_constructor_float32_600_600_3(%t168: Tensor[(600, 600, 3), float32] /* ty=Tensor[(600, 600, 3), float32] span=from_string:10752:5 */) => {
      %t168
    },
  }
}

def @tl[A](%xs13: List[A] /* ty=List[A] span=from_string:10775:11 */) -> List[A] {
  match? (%xs13) {
    Cons(_, %rest6: List[A] /* ty=List[A] span=from_string:10776:5 */) => {
      %rest6
    },
  }
}

def @update[A](%xs14: List[A] /* ty=List[A] span=from_string:10790:17 */, %n2: int32 /* ty=int32 span=from_string:10789:22 */, %v46: A /* ty=A span=from_string:10791:35 */) -> List[A] {
  %36 = equal(%n2, 0 /* ty=int32 span=from_string:10783:23 */) /* ty=bool span=from_string:10784:7 */;
  if (%36) {
    %37 = @tl(%xs14) /* ty=List[A] span=from_string:10786:16 */;
    Cons(%v46, %37) /* ty=List[A] span=from_string:10785:5 */
  } else {
    %38 = @tl(%xs14) /* ty=List[A] span=from_string:10791:21 */;
    %39 = subtract(%n2, 1 /* ty=int32 span=from_string:10789:28 */) /* ty=int32 span=from_string:10791:28 */;
    %40 = @hd(%xs14) /* ty=A span=from_string:10792:10 */;
    %41 = @update(%38, %39, %v46) /* ty=List[A] span=from_string:10792:17 */;
    Cons(%40, %41) /* ty=List[A] span=from_string:10788:5 */
  }
}

#[metadata]
{
  "root": 1,
  "nodes": [
    {
      "type_key": ""
    },
    {
      "type_key": "Map",
      "keys": [
        "relay.Constant"
      ],
      "data": [2]
    },
    {
      "type_key": "Array",
      "data": [3]
    },
    {
      "type_key": "relay.Constant",
      "attrs": {
        "_checked_type_": "8",
        "data": "0",
        "span": "6",
        "virtual_device_": "4"
      }
    },
    {
      "type_key": "VirtualDevice",
      "attrs": {
        "device_type_int": "-1",
        "memory_scope": "5",
        "target": "0",
        "virtual_device_id": "-1"
      }
    },
    {
      "type_key": "runtime.String"
    },
    {
      "type_key": "Span",
      "attrs": {
        "column": "62",
        "end_column": "81",
        "end_line": "1465",
        "line": "1465",
        "source_name": "7"
      }
    },
    {
      "type_key": "SourceName",
      "repr_str": "from_string"
    },
    {
      "type_key": "relay.TensorType",
      "attrs": {
        "dtype": "int32",
        "shape": "9",
        "span": "0"
      }
    },
    {
      "type_key": "Array",
      "data": [10]
    },
    {
      "type_key": "IntImm",
      "attrs": {
        "dtype": "int32",
        "span": "0",
        "value": "1"
      }
    }
  ],
  "b64ndarrays": [
    "P6G0lvBAXt0AAAAAAAAAAAEAAAAAAAAAAQAAAAAgAQABAAAAAAAAAAQAAAAAAAAAAAAAAA=="
  ],
  "attrs": {"tvm_version": "0.9.dev0"}
}
"""

checked_module = """
#[version = "0.0.5"]
def @main(%image_tensor: Tensor[(1, 512, 512, 3), uint8] /* ty=Tensor[(1, 512, 512, 3), uint8] span=from_string:385:13 */) {
  %0 = cast(%image_tensor, dtype="float32") /* ty=Tensor[(1, 512, 512, 3), float32] span=from_string:387:50 */;
  %1 = squeeze(%0, axis=[0]) /* ty=Tensor[(512, 512, 3), float32] */;
  %2 = expand_dims(%1, axis=0) /* ty=Tensor[(1, 512, 512, 3), float32] span=from_string:392:24 */;
  %3 = image.resize2d(%2, size=[600, 600], roi=[0f, 0f, 0f, 0f], layout="NHWC", coordinate_transformation_mode="asymmetric", rounding_method="") /* ty=Tensor[(1, 600, 600, 3), float32] span=from_string:393:16 */;
  squeeze(%3, axis=[0]) /* ty=Tensor[(600, 600, 3), float32] span=from_string:395:46 */
}
"""


# Pytest Specific Function
def test_pass_eliminate_tensor_array_ops():
    mod = tvm.relay.fromtext(ADT_module)
    passes = [compass_transform.EliminateTensorArrayOp()]
    with tvm.transform.PassContext(opt_level=3):
        update_mod = tvm.transform.Sequential(passes)(mod)
    checked_mod = tvm.relay.fromtext(checked_module)
    assert tvm.ir.structural_equal(update_mod["main"], checked_mod["main"])
    inp = np.random.rand(1, 512, 512, 3) * 255
    inp = inp.astype(np.uint8)
    check_func = relay.create_executor(mod=checked_mod, device=tvm.cpu(0), target="llvm").evaluate()
    update_func = relay.create_executor(mod=update_mod, device=tvm.cpu(0), target="llvm").evaluate()
    check_out = check_func(inp)
    update_out = update_func(inp)
    tvm.testing.assert_allclose(check_out.numpy(), update_out.numpy())


if __name__ == "__main__":
    test_pass_eliminate_tensor_array_ops()
