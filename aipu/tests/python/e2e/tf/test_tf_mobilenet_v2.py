# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2023-2024 Arm Technology (China) Co. Ltd.
import pytest
import numpy as np
from tvm.relay.backend.contrib.aipu_compass import AipuCompass
from tvm.relay.backend.contrib.aipu_compass import testing as aipu_testing
from tvm.relay.backend.contrib.aipu_compass.testing.gen_model_inputs import ImageNetVal


def run_mobilenet(model, runtime="sim", is_topk=False, threshold_cos=0.97, threshold_topk=None):
    # 1. Create AIPU Compass instance and set configurations.
    cfg_path = f"{aipu_testing.DATA_DIR}/tf_{model}.cfg"
    compass = AipuCompass(cfg_path)

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
    image = aipu_testing.get_imagenet_input(preprocess_mode="tf_inc")
    outputs = ee.run(image)

    # 5. Check result.
    predictions = outputs[0].numpy()[0]
    top3_idxes = np.argsort(predictions)[-3:][::-1]
    synset = aipu_testing.get_imagenet_synset(1001)
    print(f"Top3 labels: {[synset[idx] for idx in top3_idxes]}")
    print(f"Top3 predictions: {[predictions[idx] for idx in top3_idxes]}")

    # 6. Check cosin distance
    tf_model = aipu_testing.TFModel(cfg_path)
    aipu_testing.get_test_result(tf_model, image, outputs, threshold_cos, runtime=runtime)

    # 7. Check topk
    if is_topk:
        aipu_testing.get_topk_result(
            ee,
            f"tf_{model}",
            ImageNetVal(),
            10,
            threshold=threshold_topk,
            im_height=224,
            im_width=224,
            preprocess_mode="tf_inc",
            runtime=runtime,
        )


@pytest.mark.parametrize("runtime", ("rpc", "sim"))
@aipu_testing.clear_traceback
def test_mobilenet_v2(runtime):
    run_mobilenet("mobilenet_v2", runtime=runtime)


if __name__ == "__main__":
    test_mobilenet_v2("sim")
