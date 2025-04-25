# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2023-2024 Arm Technology (China) Co. Ltd.
"""Provide the data and functions required by testing."""
from tvm.aipu.utils import get_rpc_session
from .testing import (
    DATA_DIR,
    get_test_result,
    compare_with_gt,
    calc_mean_ap,
    get_topk_result,
    FRAMEWORK_LIST,
    clear_traceback,
    calc_mean_iou,
    write_result_to_file,
    calc_metric,
    get_tvm_output,
    compare_relax_result,
    compare_relax_opt_float_result,
    DEVICE_COMPILER,
    get_output_dict,
    calc_l1_norm,
    calc_l1_norm_with_golden,
    calc_cos_distance,
)
from .gen_model_inputs import (
    get_imagenet_input,
    get_imagenet_synset,
    get_real_image,
    yolo_v3_preprocess,
    yolo_v3_608_preprocess,
    yolo_v2_preprocess,
    yolo_v4_preprocess,
    ssd_mobilenet_preprocess,
    ssd_resnet_preprocess,
    laneaf_preprocess,
)
from .model_forward_engine import (
    TFModel,
    TFLiteModel,
    CaffeModel,
    ONNXModel,
    DarknetModel,
    TorchModel,
    PaddleModel,
    RelaxModel,
)
from .model_cfg_builder import (
    gen_model_name,
    gen_dim_info,
    get_input_tensor_of_tf,
    gen_conv_params,
    get_pool_out_shapes,
    skip_case,
    get_conv_in_out_shapes,
    get_model_cfg_path,
    is_model_file_exists,
)
from .gen_op_inputs import (
    get_op_input,
    ONNX_DTYPE_MAPPING,
)
