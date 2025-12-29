# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2023-2025 Arm Technology (China) Co. Ltd.
import os
import pytest
from tvm.compass.relax import Compass, CompassConfig, testing


@pytest.mark.X1
@pytest.mark.parametrize("gb_profile", ["true", "false"])
def test_profiler(gb_profile):
    os.environ["CPS_TVM_GBUILDER_PROFILE"] = gb_profile
    try:
        cfg = f"{testing.DATA_DIR}/relax_tiny.cfg"
        # 1. Create Compass instance and set configurations.
        compass = Compass(cfg)

        # 2. Compile the nn model.
        deployable = compass.compile(target="llvm")

        # 3. Create execution engine.
        ee = deployable.create_execution_engine()

        # 4. Run the nn model.
        preprocessed_image = testing.get_imagenet_input(preprocess_mode="tf_inc")
        ee.run(preprocessed_image)

        # 5. Check the profile.
        local_output_dir = CompassConfig.get().common["output_dir"]
        runtime_work_dir = os.path.join(local_output_dir, "tvm_compass_subfunc0", "runtime")

        aiff_dumps_dir = runtime_work_dir + "/aiff_dumps"
        cnt = len([x for x in os.scandir(aiff_dumps_dir) if x.is_dir()]) if os.path.exists(aiff_dumps_dir) else 0
        assert (cnt != 0) if gb_profile == "true" else (cnt == 0)

    finally:
        del os.environ["CPS_TVM_GBUILDER_PROFILE"]


if __name__ == "__main__":
    test_profiler("true")
    test_profiler("false")
