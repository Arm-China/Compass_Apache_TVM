# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2023-2025 Arm Technology (China) Co. Ltd.
import pytest
from tvm.compass.relax import Compass, testing


@pytest.mark.REQUIRE_RPC
@testing.clear_traceback
def test_profile_executor():
    cfg = f"{testing.DATA_DIR}/relax_tiny.cfg"
    # 1. Build NN model for the remote device.
    # Create Compass instance and set configurations.
    compass = Compass(cfg)
    # Compile the NN model.
    deployable = compass.compile(target="llvm -mtriple=aarch64-linux-gnu")

    # 2. Export and upload the deployable directory to remote device.
    rpc_sess = testing.get_rpc_session()
    device_compiler = testing.DEVICE_COMPILER
    assert device_compiler is not None, "need to set device cross compiler path."
    ee = deployable.create_execution_engine(rpc_sess, cc=device_compiler, with_profile=True)

    # 3. Profile the compiled NN model on remote device through RPC.
    preprocessed_image = testing.get_imagenet_input(preprocess_mode="tf_inc")
    print(ee.profile(preprocessed_image))


if __name__ == "__main__":
    test_profile_executor()
