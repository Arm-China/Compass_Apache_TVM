# This file is CONFIDENTIAL and created by Arm Technology (China) Co., Ltd.
# See the copyright file distributed with this work for additional information
# regarding copyright ownership.
import os
import numpy as np
from tvm import contrib
from tvm.relay.backend.contrib.aipu_compass import AipuCompass, ExecutionEngine
from tvm.relay.backend.contrib.aipu_compass import testing as aipu_testing


CURRENT_FILE_DIR = os.path.dirname(os.path.abspath(__file__))


def _run_relay_tiny(test_name):
    cfg_path = f"{aipu_testing.DATA_DIR}/relay_tiny.cfg"

    # 1. Build stages run on host development environment.
    # Create AIPU Compass instance and set configurations.
    compass = AipuCompass(cfg_path)
    # Compile the nn model.
    deployable = compass.compile()

    if test_name == "export_and_load":
        # Export the deployable file.
        tmp_dir = contrib.utils.tempdir()
        deploy_file_path = tmp_dir.relpath("nn_model.so")
        deployable.export(deploy_file_path)

        # 2. Execute stages run on target device environment.
        # Create execution engine with the deployable file and run the nn model.
        ee = ExecutionEngine(deploy_file_path)
    else:
        # Create default execution engine
        ee = deployable.create_execution_engine()

    preprocessed_image = aipu_testing.get_imagenet_input(preprocess_mode="tf_inc")
    outputs = ee.run(preprocessed_image)

    # 3. Check correctness for regression test.
    # Check cosine distance with the outputs of original framework.
    tvm_model = aipu_testing.RelayModel(cfg_path)
    aipu_testing.get_test_result(tvm_model, preprocessed_image, outputs)

    if test_name == "cpp_deploy":
        preprocessed_image.tofile(CURRENT_FILE_DIR + "/cpp/deploy/input.bin")
        deployable.export(CURRENT_FILE_DIR + "/cpp/deploy/model.so")
    return outputs[0].numpy()[0].flatten()


# Pytest Specific Function
def test_vm_export_and_load():
    os.environ["AIPU_TVM_EXECUTOR"] = "vm"
    try:
        _run_relay_tiny("export_and_load")
    finally:
        del os.environ["AIPU_TVM_EXECUTOR"]


# Pytest Specific Function
def test_graph_export_and_load():
    os.environ["AIPU_TVM_EXECUTOR"] = "graph"
    try:
        _run_relay_tiny("export_and_load")
    finally:
        del os.environ["AIPU_TVM_EXECUTOR"]


# Pytest Specific Function
def test_cpp_deploy():
    # get python output
    output_py = _run_relay_tiny("cpp_deploy")

    # get cpp output
    cwd = os.getcwd()
    os.chdir(CURRENT_FILE_DIR + "/cpp/deploy")
    os.system("mkdir build")
    os.chdir("build")
    os.system("cmake .. && make")
    os.system("./cpp_deploy ../model.so ../input.bin")
    os.chdir(cwd)
    output_cpp = np.fromfile(CURRENT_FILE_DIR + "/cpp/deploy/build/output.bin", dtype="float32")

    np.testing.assert_array_equal(output_py, output_cpp)


if __name__ == "__main__":
    test_vm_export_and_load()
    test_graph_export_and_load()
    test_cpp_deploy()
