# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2023-2024 Arm Technology (China) Co. Ltd.
import os
import pytest
import numpy as np
import random
from tvm.relay.backend.contrib.aipu_compass import AipuCompass
from tvm.relay.backend.contrib.aipu_compass import testing as aipu_testing


def run_erfnet(runtime, img_num=10):
    cfg = f"{aipu_testing.DATA_DIR}/caffe_erfnet.cfg"
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

    # 4. Run and check result.
    # a. Prepare testing dataset.
    dataset_home = os.environ["ZHOUYI_MODEL_ZOO_HOME"]
    data_file = f"{dataset_home}/caffe_erfnet/data.npy"
    dataset = np.load(data_file)

    # b. Check cosine distance with original framework on simulator.
    if runtime == "sim":
        image = np.transpose(np.expand_dims(dataset[0], axis=0), [0, 3, 1, 2]).astype("float32")
        outputs = ee.run(image)
        caffe_model = aipu_testing.CaffeModel(cfg)
        aipu_testing.get_test_result(caffe_model, image, outputs, 0.9, runtime=runtime)
        return

    # c. Check metric.
    label_file = f"{dataset_home}/caffe_erfnet/label.npy"
    labels = np.load(label_file)
    total_img_num = 100
    class_num = 19
    random.seed(1)
    img_ids = random.sample(list(range(total_img_num)), img_num)
    predicts = []
    targets = []
    for img_id in img_ids:
        image = dataset[img_id]
        image = np.expand_dims(image, axis=0)
        image = np.transpose(image, [0, 3, 1, 2]).astype("float32")
        outputs = ee.run(image)
        predict = outputs[0].numpy()
        predict = predict.argmax(axis=1)[0]
        predicts.append(predict)
        targets.append(labels[img_id])
    mIoU = aipu_testing.calc_mean_iou(predicts, targets, class_num)
    print(f"On CitySpace dataset, the mIoU is {mIoU}")
    aipu_testing.write_result_to_file(("rpc", "caffe_erfnet", "mIoU", mIoU, 0.38, "ge"))
    assert mIoU > 0.38


@pytest.mark.X2_1204
@pytest.mark.parametrize("runtime", ("rpc", "sim"))
@aipu_testing.clear_traceback
def test_erfnet(runtime):
    run_erfnet(runtime)


if __name__ == "__main__":
    test_erfnet("sim")
