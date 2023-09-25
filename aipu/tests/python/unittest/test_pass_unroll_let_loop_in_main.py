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
def @main() {
  %10 = (
    let %while_loop1: fn (int64, int64, bool, int32, Tensor[(?), int32]) -> (int64, int64, bool, int32, Tensor[(?), int32]) = fn (%i1: int64, %max_count1: int64, %cond1: bool, %prev1: int32, %range1: Tensor[(?), int32]) -> (int64, int64, bool, int32, Tensor[(?), int32]) {
      %0 = equal(%cond1, True /* ty=bool */) /* from_string */ /* ty=bool */;
      %1 = less(%i1, %max_count1) /* from_string */ /* ty=bool */;
      %2 = logical_and(%0, %1) /* from_string */ /* ty=bool */;
      if (%2) {
        %3 = copy(%prev1) /* from_string */ /* ty=int32 */;
        %4 = expand_dims(%3, axis=0) /* from_string */ /* ty=Tensor[(1), int32] */;
        %5 = (%range1, %4);
        %6 = add(%i1, 1i64 /* ty=int64 */) /* from_string */ /* ty=int64 */;
        %7 = copy(%cond1) /* from_string */ /* ty=bool */;
        %8 = add(%prev1, 1 /* ty=int32 */) /* from_string */ /* ty=int32 */;
        %9 = concatenate(%5) /* from_string */ /* ty=Tensor[(?), int32] */;
        %while_loop1(%6, %max_count1, %7, %8, %9) /* from_string */ /* ty=(int64, int64, bool, int32, Tensor[(?), int32]) */
      } else {
        (%i1, %max_count1, %cond1, %prev1, %range1)
      }
    };
    %while_loop1
  );
  %10(0i64 , 13i64 , True , 0 , meta[relay.Constant][0]) /* ty=(int64, int64, bool, int32, Tensor[(?), int32]) */
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
        "_checked_type_": "6",
        "data": "0",
        "span": "4",
        "virtual_device_": "0"
      }
    },
    {
      "type_key": "Span",
      "attrs": {
        "column": "100",
        "end_column": "121",
        "end_line": "307",
        "line": "307",
        "source_name": "5"
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
        "shape": "7",
        "span": "0"
      }
    },
    {
      "type_key": "Array",
      "data": [8]
    },
    {
      "type_key": "IntImm",
      "attrs": {
        "dtype": "int32",
        "span": "0",
        "value": "0"
      }
    }
  ],
  "b64ndarrays": [
    "P6G0lvBAXt0AAAAAAAAAAAEAAAAAAAAAAQAAAAAgAQAAAAAAAAAAAAAAAAAAAAAA"
  ],
  "attrs": {"tvm_version": "0.9.dev0"}
}
"""


# Pytest Specific Function
def test_unroll_let_loop_in_main():
    mod = tvm.relay.fromtext(ADT_module)
    passes = [compass_transform.UnrollLetLoopInMain(20), relay.transform.FoldConstant()]
    with tvm.transform.PassContext(opt_level=3):
        mod = tvm.transform.Sequential(passes)(mod)
    tuple_result = mod["main"].body
    assert isinstance(tuple_result, relay.Tuple)
    # This mod perform as a range function
    expect_result = np.array(list(range(13)), dtype=np.int32)
    tvm.testing.assert_allclose(tuple_result.fields[4].data.numpy(), expect_result, atol=0, rtol=0)


if __name__ == "__main__":
    test_unroll_let_loop_in_main()
