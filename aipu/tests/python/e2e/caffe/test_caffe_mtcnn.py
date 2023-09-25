# This file is CONFIDENTIAL and created by Arm Technology (China) Co., Ltd.
# See the copyright file distributed with this work for additional information
# regarding copyright ownership.
import os
import pytest
import numpy as np
from tvm.relay.backend.contrib.aipu_compass import AipuCompass
from tvm.relay.backend.contrib.aipu_compass import testing as aipu_testing


def run_mtcnn(model, runtime):
    cfg = f"{aipu_testing.DATA_DIR}/caffe_{model}.cfg"
    # 1. Create AIPU Compass instance and set configurations.
    compass = AipuCompass(cfg)

    # 2. Compile the nn model.
    target = "llvm -mtriple=aarch64-linux-gnu" if runtime == "rpc" else "llvm"
    deployable = compass.compile(target=target)

    # 3. Create execution engine.
    rpc_sess, device_compiler = None, None
    if runtime == "rpc":
        rpc_sess = aipu_testing.get_rpc_session()
        device_compiler = aipu_testing.DEVICE_COMPILER
        assert device_compiler is not None, "need to set device cross compiler path."
    ee = deployable.create_execution_engine(rpc_sess, cc=device_compiler)

    # 4. Run the nn model.
    dataset_home = os.environ["ZHOUYI_MODEL_ZOO_HOME"]
    data_file = f"{dataset_home}/caffe_{model}/data/0_Parade_Parade_0_102.npy"
    image = np.load(data_file, mmap_mode="c").astype(np.float32)
    outputs = ee.run(image)

    # 5. Check cosin distance
    caffe_model = aipu_testing.CaffeModel(cfg)
    aipu_testing.get_test_result(caffe_model, image, outputs, runtime=runtime)


@pytest.mark.X2_1204
@pytest.mark.parametrize("runtime", ["rpc", "simulator"])
@aipu_testing.clear_traceback
def test_mtcnn_p(runtime):
    run_mtcnn("mtcnn_p", runtime)


@pytest.mark.X2_1204
@pytest.mark.parametrize("runtime", ["rpc", "simulator"])
@aipu_testing.clear_traceback
def test_mtcnn_r(runtime):
    run_mtcnn("mtcnn_r", runtime)


@pytest.mark.X2_1204
@pytest.mark.parametrize("runtime", ["rpc", "simulator"])
@aipu_testing.clear_traceback
def test_mtcnn_o(runtime):
    run_mtcnn("mtcnn_o", runtime)


if __name__ == "__main__":
    test_mtcnn_p("simulator")
    test_mtcnn_r("simulator")
    test_mtcnn_o("simulator")
