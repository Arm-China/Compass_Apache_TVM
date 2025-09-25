# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2023-2024 Arm Technology (China) Co. Ltd.
import json
import pytest
import numpy as np
from tvm.compass.relax import Compass, CompassConfig, testing


def run_inception(model, runtime="sim"):
    cfg = f"{testing.DATA_DIR}/tf_{model}.cfg"
    # 1. Create Compass instance and set configurations.
    compass = Compass(cfg)

    # 2. Compile the nn model.
    target = "llvm -mtriple=aarch64-linux-gnu" if runtime == "rpc" else "llvm"
    deployable = compass.compile(target=target)

    # 3. Create execution engine.
    rpc_sess, device_compiler = None, None
    if runtime == "rpc":
        rpc_sess = testing.get_rpc_session()
        device_compiler = testing.DEVICE_COMPILER
        assert device_compiler is not None, "need to set device cross compiler path."
    ee = deployable.create_execution_engine(rpc_sess, cc=device_compiler)

    # 4. Run the nn model.
    _, im_height, im_width, _ = json.loads(CompassConfig.get().parser["input_shape"])
    image = testing.get_imagenet_input(im_height, im_width, preprocess_mode="tf_inc")
    outputs = ee.run(image)

    # 5. Check result.
    predictions = outputs[0].numpy()[0]
    top3_idxes = np.argsort(predictions)[-3:][::-1]
    synset = testing.get_imagenet_synset(1001)
    print(f"Top3 labels: {[synset[idx] for idx in top3_idxes]}")
    print(f"Top3 predictions: {[predictions[idx] for idx in top3_idxes]}")

    # 6. Check cosin distance
    tf_model = testing.TFModel(cfg)
    testing.get_test_result(tf_model, image, outputs, runtime=runtime)


@pytest.mark.parametrize("runtime", ("rpc", "sim"))
@testing.clear_traceback
def test_inception_v3(runtime):
    run_inception("inception_v3", runtime)


@pytest.mark.parametrize("runtime", ("rpc", "sim"))
@testing.clear_traceback
def test_inception_v4(runtime):
    run_inception("inception_v4", runtime)


@pytest.mark.parametrize("runtime", ("rpc", "sim"))
@testing.clear_traceback
def test_inception_resnet_v2(runtime):
    # After confirming by OPT topk metric, the quantization accuracy of this
    # model is ok, even through it only can get 0.96 for our current 1 image
    # cosine distance metric.
    run_inception("inception_resnet_v2", runtime)


if __name__ == "__main__":
    test_inception_v3("sim")
    test_inception_v4("sim")
    test_inception_resnet_v2("sim")
