# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2023-2024 Arm Technology (China) Co. Ltd.
import os
import pytest
from tvm.relay.backend.contrib.aipu_compass import AipuCompassConfig, AipuCompass
from tvm.relay.backend.contrib.aipu_compass import testing as aipu_testing


@pytest.mark.NOT_X3
@pytest.mark.parametrize("gb_profile", ["true", "false"])
@pytest.mark.parametrize("gb_target", ["X1_1204", "X2_1204MP3"])
def test_profiler(gb_target, gb_profile):
    os.environ["AIPU_TVM_GBUILDER_PROFILE"] = gb_profile
    os.environ["AIPU_TVM_GBUILDER_TARGET"] = gb_target
    try:
        cfg = f"{aipu_testing.DATA_DIR}/relay_tiny.cfg"
        # 1. Create AIPU Compass instance and set configurations.
        compass = AipuCompass(cfg)

        # 2. Compile the nn model.
        deployable = compass.compile(target="llvm")

        # 3. Create execution engine.
        ee = deployable.create_execution_engine()

        # 4. Run the nn model.
        preprocessed_image = aipu_testing.get_imagenet_input(preprocess_mode="tf_inc")
        ee.run(preprocessed_image)

        # 5. Check the profile.
        local_output_dir = AipuCompassConfig.get().common["output_dir"]
        runtime_work_dir = os.path.join(local_output_dir, "tvmgen_default_aipu_compass_main_0", "runtime")

        aiff_dumps_dir = runtime_work_dir + "/aiff_dumps"
        if gb_profile == "true":
            assert len(os.listdir(aiff_dumps_dir)) != 0
        else:
            if os.path.exists(aiff_dumps_dir):
                assert len(os.listdir(aiff_dumps_dir)) == 0

    finally:
        del os.environ["AIPU_TVM_GBUILDER_PROFILE"]
        del os.environ["AIPU_TVM_GBUILDER_TARGET"]


if __name__ == "__main__":
    test_profiler("X2_1204MP3", "true")
    test_profiler("X2_1204MP3", "false")
    test_profiler("X1_1204", "true")
    test_profiler("X1_1204", "false")
