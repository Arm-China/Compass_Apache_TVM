# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2023-2024 Arm Technology (China) Co. Ltd.
import os
from tvm.compass.relax import Compass, CompassConfig, testing


def run_model():
    cfg = f"{testing.DATA_DIR}/relax_tiny.cfg"
    # 1. Create Compass instance and set configurations.
    compass = Compass(cfg)

    # 2. Compile the nn model.
    deployable = compass.compile(target="llvm")

    # 3. Create execution engine.
    ee = deployable.create_execution_engine()

    # 4. Run the nn model.
    image = testing.get_imagenet_input(preprocess_mode="tf_inc")
    ee.run(image)

    # 5. Check the dump function.
    local_output_dir = CompassConfig.get().common["output_dir"]
    gbuilder_work_dir = os.path.join(local_output_dir, "tvm_compass_subfunc0", "gbuilder")

    is_ins_dumped = os.path.isfile(gbuilder_work_dir + "/input0_int8.bin")
    is_outs_dumped = os.path.isfile(gbuilder_work_dir + "/output0_int8.bin")

    if os.getenv("CPS_TVM_RUNTIME_DUMP") == "True":
        assert is_ins_dumped, "Expect inputs dumped."
        assert is_outs_dumped, "Expect outputs dumped."
    else:
        assert not is_ins_dumped, "Expect inputs not dumped."
        assert not is_outs_dumped, "Expect outputs not dumped."


def test_not_dump():
    run_model()


def test_dump():
    os.environ["CPS_TVM_RUNTIME_DUMP"] = "True"
    try:
        run_model()
    finally:
        del os.environ["CPS_TVM_RUNTIME_DUMP"]


if __name__ == "__main__":
    test_not_dump()
    test_dump()
