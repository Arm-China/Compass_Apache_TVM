# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2023-2025 Arm Technology (China) Co. Ltd.
import os
import pytest
from tvm.compass.relax import Compass, CompassConfig, testing


def run_spm():
    os.environ["CPS_TVM_RUNTIME_SPM"] = "1000;300;50;512;0.8;html"
    try:
        cfg = f"{testing.DATA_DIR}/relax_tiny.cfg"
        # 1. Create Compass instance and set configurations.
        compass = Compass(cfg)

        # 2. Compile the nn model.
        deployable = compass.compile(target="llvm")

        # 3. Create execution engine.
        ee = deployable.create_execution_engine()

        # 4. Run the nn model.
        preprocessed_image = testing.get_imagenet_input(preprocess_mode="tf_inc")
        ee.run(preprocessed_image)

        # 5. Check the report file.
        output_dir = CompassConfig.get().common["output_dir"]
        assert os.path.exists(f"{output_dir}/tvm_compass_subfunc0/runtime/perf.html")
    finally:
        del os.environ["CPS_TVM_RUNTIME_SPM"]


@pytest.mark.NOT_X1
@pytest.mark.NOT_X2
def test_spm():
    if "CPS_TVM_GBUILDER_TARGET" in os.environ:
        run_spm()
        return

    os.environ["CPS_TVM_GBUILDER_TARGET"] = "X3P_1304"
    try:
        run_spm()
    finally:
        del os.environ["CPS_TVM_GBUILDER_TARGET"]


if __name__ == "__main__":
    test_spm()
