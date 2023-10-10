# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2023 Arm Technology (China) Co. Ltd.
import tvm
from tvm.relay.backend.contrib.aipu_compass import analysis as compass_analysis


def test_check_bellwether():
    # 1. Multiple definition.
    ir_mod = tvm.relay.fromtext(
        """
#[version = "0.0.5"]
def @my_add1(%i2: int32, %i3: int32, Compiler="byoc", global_symbol="my_add1") -> int32 {
  add(%i2, %i3)
}

def @my_add2(%i2: int32, %i3: int32, Compiler="byoc", global_symbol="my_add2") -> int32 {
  add(%i2, %i3)
}

def @main(%i0: int32, %i1: int32) -> int32 {
  %0 = @my_add1(%i0, %i1);
  @my_add2(%0, %i1)
}
"""
    )
    assert compass_analysis.check_bellwether(ir_mod, "byoc") is False

    # 2. Multiple call.
    ir_mod = tvm.relay.fromtext(
        """
#[version = "0.0.5"]
def @my_add(%i2: int32, %i3: int32, Compiler="byoc", global_symbol="my_add") -> int32 {
  add(%i2, %i3)
}

def @main(%i0: int32, %i1: int32) -> int32 {
  %0 = @my_add(%i0, %i1);
  @my_add(%0, %i1)
}
"""
    )
    assert compass_analysis.check_bellwether(ir_mod, "byoc") is False

    # 3. Parameter and argument have different length.
    ir_mod = tvm.relay.fromtext(
        """
#[version = "0.0.5"]
def @my_add(%i1: int32, %i2: int32, Compiler="byoc", global_symbol="my_add") -> int32 {
  add(%i1, %i2)
}

def @main(%i0: int32) -> int32 {
  @my_add(%i0, 3)
}
"""
    )
    assert compass_analysis.check_bellwether(ir_mod, "byoc") is False

    # 4. Parameter and argument are not same.
    ir_mod = tvm.relay.fromtext(
        """
#[version = "0.0.5"]
def @my_add(%i2: int32, %i3: int32, Compiler="byoc", global_symbol="my_add") -> int32 {
  add(%i2, %i3)
}

def @main(%i0: int32, %i1: int32) -> int32 {
  @my_add(%i0, 3)
}
"""
    )
    assert compass_analysis.check_bellwether(ir_mod, "byoc") is False

    # 5. Positive case.
    ir_mod = tvm.relay.fromtext(
        """
#[version = "0.0.5"]
def @my_add(%i2: int32, %i3: int32, Compiler="byoc", global_symbol="my_add") -> int32 {
  add(%i2, %i3)
}

def @main(%i0: int32, %i1: int32) -> int32 {
  @my_add(%i0, %i1)
}
"""
    )
    assert compass_analysis.check_bellwether(ir_mod, "byoc") is True


if __name__ == "__main__":
    test_check_bellwether()
