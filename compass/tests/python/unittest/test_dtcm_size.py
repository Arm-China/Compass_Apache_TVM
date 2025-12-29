# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2023-2025 Arm Technology (China) Co. Ltd.
import os
import glob
import pytest
from tvm import contrib
from tvm.compass.relax import Compass, CompassConfig, ExecutionEngine, testing


@pytest.mark.X1
def test_dtcm_size():
    os.environ["CPS_TVM_GBUILDER_TCM_SIZE"] = "2048"
    try:
        cfg_path = f"{testing.DATA_DIR}/relax_tiny.cfg"

        # 1. Build stages run on host development environment.
        # Create Compass instance and set configurations.
        compass = Compass(cfg_path)
        # Compile the nn model.
        deployable = compass.compile()

        # Export the deployable directory.
        tmp_dir = contrib.utils.tempdir()
        deploy_file_path = tmp_dir.relpath("nn_model.so")
        deployable.export(deploy_file_path)

        # 2. Execute stages run on target device environment.
        # Create execution engine with the deployable directory and run the NN model.
        ee = ExecutionEngine(deploy_file_path)
        preprocessed_image = testing.get_imagenet_input(preprocess_mode="tf_inc")
        ee.run(preprocessed_image)

        # 3. Check correctness for regression test.
        output_dir = CompassConfig.get().common["output_dir"]
        sim_cfg = open(glob.glob(f"{output_dir}/*/runtime/runtime.cfg")[0]).read().split()
        expect = "DTCM_SIZE=0x200000"
        assert expect in sim_cfg, f"\nExpect snippet:\n{expect}\n\nRuntime config:\n{sim_cfg}\n"

    finally:
        del os.environ["CPS_TVM_GBUILDER_TCM_SIZE"]


if __name__ == "__main__":
    test_dtcm_size()
