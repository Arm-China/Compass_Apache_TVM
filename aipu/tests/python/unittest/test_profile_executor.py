# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2023-2024 Arm Technology (China) Co. Ltd.
import os
from tvm.relay.backend.contrib.aipu_compass import AipuCompass
from tvm.relay.backend.contrib.aipu_compass import testing as aipu_testing


def run_profile_executor():
    cfg = f"{aipu_testing.DATA_DIR}/relay_tiny.cfg"
    # 1. Build NN model for the remote device.
    # Create AIPU Compass instance and set configurations.
    compass = AipuCompass(cfg)
    # Compile the NN model.
    deployable = compass.compile(target="llvm -mtriple=aarch64-linux-gnu")

    # 2. Export and upload the deployable file to remote device.
    rpc_sess = aipu_testing.get_rpc_session()
    device_compiler = aipu_testing.DEVICE_COMPILER
    assert device_compiler is not None, "need to set device cross compiler path."
    ee = deployable.create_execution_engine(rpc_sess, cc=device_compiler, with_profile=True)

    # 3. Profile the compiled NN model on remote device through RPC.
    preprocessed_image = aipu_testing.get_imagenet_input(preprocess_mode="tf_inc")
    print(ee.profile(preprocessed_image))


def test_profile_vm():
    os.environ["AIPU_TVM_EXECUTOR"] = "vm"
    try:
        run_profile_executor()
    finally:
        del os.environ["AIPU_TVM_EXECUTOR"]


def test_profile_graph():
    os.environ["AIPU_TVM_EXECUTOR"] = "graph"
    try:
        run_profile_executor()
    finally:
        del os.environ["AIPU_TVM_EXECUTOR"]


if __name__ == "__main__":
    test_profile_vm()
    test_profile_graph()
