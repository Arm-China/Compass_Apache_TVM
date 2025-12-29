# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2023-2025 Arm Technology (China) Co. Ltd.
import os
import numpy as np
from tvm.compass.relax import Compass, CompassConfig, testing


def run_model(tiling_cfg, runtime="sim"):
    # 1. Create Compass instance and set configurations.
    cfg = """
    [Common]
    [Parser]
    model_type = tensorflow
    model_name = mobilenet_v2
    input_model = ${ZHOUYI_MODEL_ZOO_HOME}/tf_mobilenet_v2/mobilenet_v2.pb
    input = input
    input_shape = [1,224,224,3]
    output = MobilenetV2/Predictions/Reshape_1:0
    [Optimizer]
    calibration_batch_size = 20
    weight_bits = 8
    bias_bits = 32
    activation_bits = 8
    dataset = NumpyDataset
    calibration_data = ${ZHOUYI_MODEL_ZOO_HOME}/tf_mobilenet_v2/calibration_data.npy
    [GBuilder]
    target = X2_1204MP3
    """
    cfg = cfg + tiling_cfg + "\n"
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

    # 4. Run and check result.
    image = testing.get_imagenet_input(preprocess_mode="tf_inc")
    outputs = ee.run(image)

    # Check result.
    predictions = outputs[0].numpy()[0]
    top3_idxes = np.argsort(predictions)[-3:][::-1]
    synset = testing.get_imagenet_synset(1001)
    print(f"Top3 labels: {[synset[idx] for idx in top3_idxes]}")
    print(f"Top3 predictions: {[predictions[idx] for idx in top3_idxes]}")

    # Check cosin distance
    tflite_model = testing.TFModel(cfg)
    testing.get_test_result(tflite_model, image, outputs, runtime=runtime, threshold=0.97)

    expect = None
    if tiling_cfg == "":
        return
    elif "fps" in tiling_cfg:
        expect = "tiling: fps"
    elif "footprint" in tiling_cfg:
        expect = "tiling: footprint"

    local_output_dir = CompassConfig.get().common["output_dir"]
    log_path = os.path.join(local_output_dir, "tvm_compass_subfunc0", "gbuilder", "aipugb.log")

    log_str = open(log_path, "r").read()
    msg = f"\nExpect snippet:\n{expect}\n\nCompass GBuilder Log:\n{log_str}\n"
    assert expect in log_str, msg


def test_gb_tiling():
    run_model("")
    run_model("tiling = footprint")
    run_model("tiling = fps")


if __name__ == "__main__":
    test_gb_tiling()
