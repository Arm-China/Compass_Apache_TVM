# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2023-2024 Arm Technology (China) Co. Ltd.
import os
from tvm.relay.backend.contrib.aipu_compass import AipuCompass, AipuCompassConfig
from tvm.relay.backend.contrib.aipu_compass import testing as aipu_testing


def run_model():
    cfg = f"{aipu_testing.DATA_DIR}/relay_tiny.cfg"
    # 1. Create AIPU Compass instance and set configurations.
    compass = AipuCompass(cfg)

    # 2. Compile the nn model.
    deployable = compass.compile(target="llvm")

    # 3. Create execution engine.
    ee = deployable.create_execution_engine()

    # 4. Run the nn model.
    preprocessed_image = aipu_testing.get_imagenet_input(preprocess_mode="tf_inc")
    ee.run(preprocessed_image)

    # 5. Check the dump function.
    local_output_dir = AipuCompassConfig.get().common["output_dir"]
    gbuilder_work_dir = os.path.join(local_output_dir, "tvmgen_default_aipu_compass_main_0", "gbuilder")

    is_ins_dumped = os.path.isfile(gbuilder_work_dir + "/input0_int8.bin")
    is_outs_dumped = os.path.isfile(gbuilder_work_dir + "/output0_int8.bin")

    if os.getenv("AIPU_TVM_RUNTIME_DUMP") == "True":
        assert is_ins_dumped, "Expect inputs dumped."
        assert is_outs_dumped, "Expect outputs dumped."
    else:
        assert not is_ins_dumped, "Expect inputs not dumped."
        assert not is_outs_dumped, "Expect outputs not dumped."


def test_not_dump():
    run_model()


def test_dump():
    os.environ["AIPU_TVM_RUNTIME_DUMP"] = "True"
    try:
        run_model()
    finally:
        del os.environ["AIPU_TVM_RUNTIME_DUMP"]


if __name__ == "__main__":
    test_not_dump()
    test_dump()
