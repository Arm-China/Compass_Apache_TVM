# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2023-2024 Arm Technology (China) Co. Ltd.
import os
import pytest
import glob
from tvm import contrib
from tvm.compass.relax import Compass, CompassConfig, ExecutionEngine, testing
from tvm.compass.utils import sync_compass_output_dir


def _run_rpc(is_one_stage):
    cfg_path = f"{testing.DATA_DIR}/relax_tiny.cfg"

    # 1. Build NN model for the remote device.
    # Create Compass instance and set configurations.
    compass = Compass(cfg_path)
    # Compile the nn model.
    deployable = compass.compile(target="llvm -mtriple=aarch64-linux-gnu")

    # 2. Export and upload the deployable directory to remote device.
    rpc_sess = testing.get_rpc_session()
    device_compiler = testing.DEVICE_COMPILER
    assert device_compiler is not None, "need to set device cross compiler path."

    if is_one_stage:
        ee = deployable.create_execution_engine(rpc_sess, cc=device_compiler)
    else:
        # The 1st stage for exporting the deployable directory.
        tmp_dir = contrib.utils.tempdir()
        deploy_dir_path = tmp_dir.relpath("test_rpc_two_stage")
        deployable.export(deploy_dir_path, cc=device_compiler)
        # The 2nd stage for creating execution engine using the exported deployable directory through RPC.
        ee = ExecutionEngine(deploy_dir_path, rpc_sess)

    # 3. Run the compiled NN model on remote device through RPC.
    preprocessed_image = testing.get_imagenet_input(preprocess_mode="tf_inc")
    outputs = ee.run(preprocessed_image)

    # 4. Check correctness for regression test.
    # Check cosine distance with the outputs of original framework.
    tvm_model = testing.RelaxModel(cfg_path)
    testing.get_test_result(tvm_model, preprocessed_image, outputs)

    # Check profiler.
    sync_compass_output_dir(rpc_sess)


@pytest.mark.REQUIRE_RPC
@testing.clear_traceback
@pytest.mark.parametrize("is_one_stage,", (True, False))
def test_rpc_stage(is_one_stage):
    _run_rpc(is_one_stage)


@pytest.mark.REQUIRE_RPC
@testing.clear_traceback
def test_profiler():
    os.environ["CPS_TVM_GBUILDER_PROFILE"] = "true"
    try:
        _run_rpc(True)
        local_output_dir = CompassConfig.get().common["output_dir"]
        assert len(glob.glob(f"{local_output_dir}/*/runtime/profile_data.bin")) != 0
    finally:
        del os.environ["CPS_TVM_GBUILDER_PROFILE"]


if __name__ == "__main__":
    test_rpc_stage(True)
    test_rpc_stage(False)
    test_profiler()
