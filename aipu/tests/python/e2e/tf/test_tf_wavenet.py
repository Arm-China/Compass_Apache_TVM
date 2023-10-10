# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2023 Arm Technology (China) Co. Ltd.
import pytest
from tvm.relay.backend.contrib.aipu_compass import AipuCompass, AipuCompassConfig
from tvm.relay.backend.contrib.aipu_compass import testing as aipu_testing

import torch


def run_wavenet(model, runtime):
    cfg = f"{aipu_testing.DATA_DIR}/tf_{model}.cfg"
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

    # 4. Calcu metric result.
    opt_cfg = AipuCompassConfig.get().optimizer
    runtime_cfg = AipuCompassConfig.get().runtime

    post_process = lambda x: torch.where(x == 65535, torch.IntTensor([-1]), x)
    opt_result = aipu_testing.calc_metric(
        opt_cfg["dataset"],
        runtime_cfg["data"],
        runtime_cfg["label"],
        runtime_cfg["metric"],
        10,
        ee,
        post_process,
    )
    aipu_testing.write_result_to_file((runtime, f"tf_{model}", "wermetric", opt_result, 0.21, "le"))
    assert opt_result < 0.21


@pytest.mark.X2_1204
@pytest.mark.parametrize("runtime", ["rpc", "simulator"])
@aipu_testing.clear_traceback
def test_wavenet(runtime):
    run_wavenet("wavenet", runtime)


if __name__ == "__main__":
    test_wavenet("simulator")
