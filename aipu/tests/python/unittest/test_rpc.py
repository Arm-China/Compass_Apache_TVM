# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2023-2024 Arm Technology (China) Co. Ltd.
import os
import glob
from tvm import contrib
from tvm.aipu.utils import sync_compass_output_dir
from tvm.relay.backend.contrib.aipu_compass import AipuCompass, ExecutionEngine, AipuCompassConfig
from tvm.relay.backend.contrib.aipu_compass import testing as aipu_testing


def _run_rpc(executor_name, is_one_stage):
    cfg_path = f"{aipu_testing.DATA_DIR}/relay_tiny.cfg"

    # 1. Build NN model for the remote device.
    # Create AIPU Compass instance and set configurations.
    compass = AipuCompass(cfg_path)
    # Compile the nn model.
    deployable = compass.compile(target="llvm -mtriple=aarch64-linux-gnu")

    # 2. Export and upload the deployable file to remote device.
    rpc_sess = aipu_testing.get_rpc_session()
    device_compiler = aipu_testing.DEVICE_COMPILER
    assert device_compiler is not None, "need to set device cross compiler path."

    if is_one_stage:
        ee = deployable.create_execution_engine(rpc_sess, cc=device_compiler)
    else:
        # The 1st stage for exporting the deployable file.
        tmp_dir = contrib.utils.tempdir()
        deploy_file_path = tmp_dir.relpath(f"{executor_name}_compiled_model.so")
        deployable.export(deploy_file_path, cc=device_compiler)
        # The 2nd stage for creating execution engine using the exported
        # deployable file through RPC.
        ee = ExecutionEngine(deploy_file_path, rpc_sess)

    # 3. Run the compiled NN model on remote device through RPC.
    preprocessed_image = aipu_testing.get_imagenet_input(preprocess_mode="tf_inc")
    outputs = ee.run(preprocessed_image)

    # 4. Check correctness for regression test.
    # Check cosine distance with the outputs of original framework.
    tvm_model = aipu_testing.RelayModel(cfg_path)
    aipu_testing.get_test_result(tvm_model, preprocessed_image, outputs)
    # Check performance.
    # TODO(CTV-877): In theory, set_input shouldn't be added here after
    # ee.run called, which input provided. In fact, if not added, a RPC
    # error will occur. Currently, this issue is only found on the
    # trimmed small float model.
    ee.set_inputs(preprocessed_image)
    print(ee.executor.benchmark(rpc_sess.cpu(0)))
    # Check profiler.
    sync_compass_output_dir(rpc_sess)


# Pytest Specific Function
@aipu_testing.clear_traceback
def test_rpc_one_stage_vm():
    os.environ["AIPU_TVM_EXECUTOR"] = "vm"
    try:
        _run_rpc("vm", True)
    finally:
        del os.environ["AIPU_TVM_EXECUTOR"]


# Pytest Specific Function
@aipu_testing.clear_traceback
def test_rpc_one_stage_graph():
    os.environ["AIPU_TVM_EXECUTOR"] = "graph"
    try:
        _run_rpc("graph", True)
    finally:
        del os.environ["AIPU_TVM_EXECUTOR"]


# Pytest Specific Function
@aipu_testing.clear_traceback
def test_rpc_two_stage_vm():
    os.environ["AIPU_TVM_EXECUTOR"] = "vm"
    try:
        _run_rpc("vm", False)
    finally:
        del os.environ["AIPU_TVM_EXECUTOR"]


# Pytest Specific Function
@aipu_testing.clear_traceback
def test_profiler():
    os.environ["AIPU_TVM_EXECUTOR"] = "vm"
    os.environ["AIPU_TVM_GBUILDER_PROFILE"] = "true"
    try:
        _run_rpc("vm", True)
        local_output_dir = AipuCompassConfig.get().common["output_dir"]
        assert len(glob.glob(f"{local_output_dir}/*/runtime/profile_data.bin")) != 0
    finally:
        del os.environ["AIPU_TVM_EXECUTOR"]
        del os.environ["AIPU_TVM_GBUILDER_PROFILE"]


if __name__ == "__main__":
    test_rpc_one_stage_vm()
    test_rpc_one_stage_graph()
    test_rpc_two_stage_vm()
    test_profiler()
