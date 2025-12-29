# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2023-2025 Arm Technology (China) Co. Ltd.
import os
import numpy as np
from subprocess import run
from tvm import contrib
from tvm.compass.relax import Compass, ExecutionEngine, testing


CURRENT_FILE_DIR = os.path.dirname(os.path.abspath(__file__))


def _run_relax_tiny(test_name):
    cfg_path = f"{testing.DATA_DIR}/relax_tiny.cfg"

    # 1. Build stages run on host development environment.
    # Create Compass instance and set configurations.
    compass = Compass(cfg_path)
    # Compile the nn model.
    deployable = compass.compile()

    if test_name == "export_and_load":
        # Export the deployable directory.
        tmp_dir = contrib.utils.tempdir()
        deploy_dir_path = tmp_dir.relpath("test_deploy_export_and_load")
        deployable.export(deploy_dir_path)

        # 2. Execute stages run on target device environment.
        # Create execution engine with the exported deployable directory and run the NN model.
        ee = ExecutionEngine(deploy_dir_path)
    else:
        # Create default execution engine
        ee = deployable.create_execution_engine()

    preprocessed_image = testing.get_imagenet_input(preprocess_mode="tf_inc")
    outputs = ee.run(preprocessed_image)

    # 3. Check correctness for regression test.
    # Check cosine distance with the outputs of original framework.
    tvm_model = testing.RelaxModel(cfg_path)
    testing.get_test_result(tvm_model, preprocessed_image, outputs)

    if test_name == "cpp_deploy":
        preprocessed_image.tofile(CURRENT_FILE_DIR + "/cpp/deploy/input.bin")
        deployable.export(CURRENT_FILE_DIR + "/cpp/deploy")
    return outputs[0].numpy()[0].flatten()


def test_export_and_load():
    _run_relax_tiny("export_and_load")


def test_cpp_deploy():
    # get python output
    output_py = _run_relax_tiny("cpp_deploy")

    # get cpp output
    cwd = os.getcwd()
    os.chdir(CURRENT_FILE_DIR + "/cpp/deploy")
    run(
        "rm -rf build && mkdir build && cd build && cmake .. && make && ./cpp_deploy .. ../input.bin",
        shell=True,
        check=True,
    )
    os.chdir(cwd)
    output_cpp = np.fromfile(CURRENT_FILE_DIR + "/cpp/deploy/build/output.bin", dtype="float32")

    np.testing.assert_array_equal(output_py, output_cpp)


if __name__ == "__main__":
    test_export_and_load()
    test_cpp_deploy()
