# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2023-2024 Arm Technology (China) Co. Ltd.
# pylint: disable=invalid-name
"""Check results with multiple methods."""
import os
import re
import json
import operator
import functools
from typing import List
import numpy as np
import tvm
from tvm import relax, ir
from ...logger import timer
from ..builder import create_dataset, create_metric
from ..compass import Compass
from ..config import CompassConfig, CompassFunctionConfig
from ..execution_engine import ExecutionEngine
from ..utils import compute_cos_distance
from .model_forward_engine import TFModel, TFLiteModel, ONNXModel, TorchModel, RelaxModel
from .gen_model_inputs import CoCo, get_imagenet_input, ClassificationDataset, VOC
from .common import convert_to_list, is_number, get_subgraph_and_op_count


DATA_DIR = os.path.abspath(f"{__file__}/../../../../../../compass/tests/data")
NIGHTLY_TEST_TXT = os.path.abspath(f"{__file__}/../../../../../../nightly_test_report.txt")
FRAMEWORK_LIST = ("onnx", "tf", "tflite", "torch")

AI_BENCHMARK_PERCHANNEL = {
    "unet_quant": [1, 1024, 1024, 3],
    "dped_quant": [1, 1536, 2048, 3],
    "dped_instance_quant": [1, 1024, 1536, 3],
    "xlsr_quant": [1, 1080, 1920, 3],
    "esrgan_quant": [1, 512, 512, 3],
    "srgan_quant": [1, 1024, 1024, 3],
    "pynet_quant": [1, 1024, 1024, 3],
    "vsr_quant": [1, 2160, 3840, 3],
    "imdn_quant": [1, 1024, 1024, 3],
    "punet_quant": [1, 1088, 1920, 3],
    "yolo_v4_tiny_quant": [1, 2535, 4, 1],
}
AI_BENCHMARK_NOZERO = {
    "deeplab_v3_plus_quant": [1, 256, 256, 19],
    "crnn_quant": [1, 1, 1800, 1],
    "efficientnet_b4_quant": [1, 1, 1000, 1],
    "inception_v3_quant": [1, 1, 1001, 1],
    "lstm_quant": [1, 1, 2048, 1],
    "mobilebert_quant": [1, 2, 384, 1],
    "mobilenet_v2_quant": [1, 1, 1001, 1],
    "mobilenet_v3_quant": [1, 1, 1001, 1],
    "mobilenet_v2_b8_quant": [1, 8, 1001, 1],
    "mobilenet_v3_b4_quant": [1, 4, 1001, 1],
    "mv3_depth_quant": [1, 1024, 1536, 1],
    "vgg_quant": [1, 256, 256, 1],
}
# Suppose the user doesn't use rpc, so we don't assert here if CPS_TVM_DEVICE_COMPILER is None
DEVICE_COMPILER = os.getenv("CPS_TVM_DEVICE_COMPILER")


def calc_mean_ap(
    dataset, prediction, model, framework="unknown", iou_thres=0.5, map_thres=0, runtime="sim"
):
    """
    The general function to calculate mAP@IOU0.5 based on the Dataset passed in.
    :param
    dataset  (obj)   : initilized Dataset object
    predict  (dict)  : The predict result, it should have the key
    model    (string): model name
    framework(string): framework name
    iou_thres(float) : iou threshold
    map_thres(float) : mAP threshold
    runtime  (string): Runtime of Zhouyi Compass
    :return: mAP @ IOU threshold 0.5
    """
    mean_ap = "Uncalculated"
    if isinstance(dataset, (CoCo, VOC)):
        mean_ap = calc_dataset_mean_ap(dataset, prediction, iou_thres)
    else:
        print(f"Currently, mAP calculation of {model} is not supported.")

    # A dict for storing test results, which will be written to
    # a txt file and used for email report generation.
    model_name = "_".join([framework, model])
    result_dict = {
        model_name: {
            "mAP": {
                "value": mean_ap,
                "threshold": map_thres,
            },
            "runtime": runtime,
        },
    }
    write_result_to_file(result_dict)
    assert (
        mean_ap >= map_thres
    ), f"mAP: {model_name} compare FAILED, gt:{map_thres} vs real:{mean_ap}"
    return mean_ap


def calc_dataset_mean_ap(dataset, predict, iou_thres=0.5):
    """
    calculate mAP@IOU0.5 on varies dataset
    :param
    dataset  (CoCo/Voc)     : initilized dataset object
    predict  (dict)  : The predict result, it should have the key
        id : img_id of dataset
            boxes : [[top, left, bottom, right],...] predict boxes list, shape is [N, 4]
            scores : the confidence of the boxes, shape is [N]
            categroies : predict categroies, should be converted to dataset cats_id, shape is [N]
            N is the detected bbox num on img_id images.
    :return: mAP @ IOU threshold 0.5
    """

    def _get_categories(dataset, img_ids):
        if isinstance(dataset, CoCo):
            anns = dataset.get_ann_ids(img_ids)
            categories = list({dataset.load_anns(ann_id)[0]["category_id"] for ann_id in anns})
        else:
            categories = list({ann["category_id"] for ann in dataset.load_anns(img_ids)})
        return categories

    def _get_gtbox(dataset, img_id, cat_id):
        def _map_ground_truth_box_coco(bbox):
            bbox_x, bbox_y, bbox_w, bbox_h = bbox
            return [bbox_y, bbox_x, bbox_y + bbox_h, bbox_x + bbox_w]

        def _map_ground_truth_box_voc(bbox):
            xmin, ymin, xmax, ymax = bbox
            return [ymin, xmin, ymax, xmax]

        if isinstance(dataset, CoCo):
            ann_ids = dataset.get_ann_ids(img_ids=img_id, cat_ids=cat_id)
            anns = dataset.load_anns(ann_ids)
            ground_truth_bbox = [_map_ground_truth_box_coco(ann["bbox"]) for ann in anns]
        else:
            anns = dataset.load_anns(img_ids=img_id, cat_ids=cat_id)
            ground_truth_bbox = [_map_ground_truth_box_voc(ann["bbox"]) for ann in anns]
        return ground_truth_bbox

    def _get_gtbox_nums(dataset, img_ids, cat_id):
        if isinstance(dataset, CoCo):
            anns = dataset.get_ann_ids(img_ids=img_ids, cat_ids=cat_id)
        else:
            anns = dataset.load_anns(img_ids=img_ids, cat_ids=cat_id)
        return len(anns)

    class BoxInfo:
        def __init__(self, img_id, box, score, category):
            self.img_id = int(img_id)
            self.box = box
            self.score = score
            self.category = category
            self.check_type = None

    def _calc_iou(ground_truth_box, predict_box, img_h, img_w):
        def clamp_x(value):
            return max(min(value, img_w), 0)

        def clamp_y(value):
            return max(min(value, img_h), 0)

        g_top = clamp_y(ground_truth_box[0])
        g_left = clamp_x(ground_truth_box[1])
        g_bottom = clamp_y(ground_truth_box[2])
        g_right = clamp_x(ground_truth_box[3])

        p_top = clamp_y(predict_box[0])
        p_left = clamp_x(predict_box[1])
        p_bottom = clamp_y(predict_box[2])
        p_right = clamp_x(predict_box[3])

        area_g = (g_right - g_left) * (g_bottom - g_top)
        area_p = (p_right - p_left) * (p_bottom - p_top)

        top = max(g_top, p_top)
        bottom = min(g_bottom, p_bottom)
        left = max(g_left, p_left)
        right = min(g_right, p_right)
        area = (right - left) * (bottom - top)
        area = max(0, area)

        return area / (area_g + area_p - area + 0.000001)

    def _check_bbox_type(dataset, box_infos, cat_id, thres=0.5):
        infos = [info for info in box_infos if info.category == cat_id]
        img_ids = list({info.img_id for info in infos})
        for img_id in img_ids:
            ground_truth_bbox = _get_gtbox(dataset, img_id, cat_id)
            img = dataset.load_imgs(img_id)[0]
            width, height = img["width"], img["height"]

            pred_infos = [info for info in infos if info.img_id == img_id]
            for info in pred_infos:
                if info.check_type is None:
                    for bbox in ground_truth_bbox:
                        iou = _calc_iou(bbox, info.box, height, width)
                        if iou > thres:
                            info.check_type = "TP"
                            ground_truth_bbox.remove(bbox)
                            break
                    if info.check_type is None:
                        info.check_type = "FP"

    def _voc_ap(rec, prec):
        mrec = np.concatenate(([0.0], rec, [1.0]))
        mpre = np.concatenate(([0.0], prec, [0.0]))

        for i in range(mpre.size - 1, 0, -1):
            mpre[i - 1] = np.maximum(mpre[i - 1], mpre[i])

        i = np.where(mrec[1:] != mrec[:-1])[0]

        return np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])

    def _get_precision_recall(dataset, box_infos, cat_id):
        infos = [info for info in box_infos if info.category == cat_id]
        img_ids = list({info.img_id for info in box_infos})
        ground_truth_num = _get_gtbox_nums(dataset, img_ids, cat_id)
        recall = []
        precision = []
        tp_count = 0
        for info in infos:
            if info.check_type == "TP":
                tp_count = tp_count + 1
            precision.append(tp_count / (len(precision) + 1))
            recall.append(tp_count / ground_truth_num)
        return np.array(precision), np.array(recall)

    box_infos = []
    for _id in predict.keys():
        info = predict[_id]
        scores = info["scores"]
        boxes = info["boxes"]
        classes = info["categroies"]

        for score, box, cat in zip(scores, boxes, classes):
            info = BoxInfo(_id, box, score, cat)
            box_infos.append(info)

    box_infos.sort(key=lambda info: info.score, reverse=True)
    img_ids = list({info.img_id for info in box_infos})
    categories = _get_categories(dataset, img_ids)

    aps = []
    for cat in categories:
        infos = [info for info in box_infos if info.category == cat]
        _check_bbox_type(dataset, infos, cat, iou_thres)
        precision, recall = _get_precision_recall(dataset, box_infos, cat)
        ret = _voc_ap(recall, precision)
        aps.append(ret)
    return sum(aps) / len(aps) if len(aps) > 0 else 0


def calc_mean_iou(predicts, targets, class_num):
    """
    The general function to calculate mIoU for segmentation models.
    :param
    predicts   (list)   : outputs of each image inference.
    targets    (list)   : ground truth of each image.
    class_num  (int)    : total class num.
    :return: mIoU
    """
    assert len(predicts) == len(targets)
    assert predicts[0].shape == targets[0].shape
    epsilon = 0.000000000001
    confusion_matrix = np.zeros((class_num, class_num))
    for p, g in zip(predicts, targets):
        p = p.flatten()
        g = g.flatten()
        mask = (g >= 0) & (g < class_num)
        confusion_matrix += np.bincount(
            class_num * g[mask].astype(int) + p[mask], minlength=class_num**2
        ).reshape(class_num, class_num)
    iou = np.diag(confusion_matrix) / (
        confusion_matrix.sum(0) + confusion_matrix.sum(1) - np.diag(confusion_matrix) + epsilon
    )
    return np.mean(iou)


def calc_metric(dataset, data, label, metric, max_batch, executor, post_process=None):
    """
    The general function to calculate metric using opt plugin.
    :param
    dataset         (string)            : the string of dataset.
    data            (string)            : the string of data.
    label           (string)            : the string of label.
    metric          (string)            : the string of metric.
    max_batch       (int)               : the max time will be run.
    executor        (ExecutionEngine)   : ExecutionEngine that run with input.
    post_process    (function)          : deal with the output.
    :return: metric result.
    """
    # pylint: disable=import-outside-toplevel
    from torch.utils.data.dataloader import DataLoader
    from torch import from_numpy

    def _to_input(data):
        data = data.numpy()
        if data.dtype == np.float64:
            data = data.astype(np.float32)
        return data

    def _to_torch(data):
        if isinstance(data, ir.Array):
            res = []
            for i in data:
                res.append(_to_torch(i))
            return res
        if not isinstance(data, np.ndarray):
            # Assume data is ndarray.
            return from_numpy(data.numpy())
        return from_numpy(data)

    data = os.path.abspath(os.path.expanduser(os.path.expandvars(data)))
    label = os.path.abspath(os.path.expanduser(os.path.expandvars(label)))

    dataset = create_dataset(dataset, data, label)
    dataloader = DataLoader(dataset)
    metric = metric.split("(", 1)
    class_name = metric[0].strip()
    args = metric[1] if len(metric) == 2 else ""
    args = [x.strip() for x in args.strip(")").split(",") if x.strip() != ""]
    metric = create_metric(class_name, *args)

    cur_batch = 0
    for inp, target in dataloader:
        if cur_batch >= max_batch:
            break

        input_data = []
        if not isinstance(inp, list):
            inp = [inp]
        for inp_data in inp:
            input_data.append(_to_input(inp_data))

        outputs = executor.run(*input_data)
        output = _to_torch(outputs)

        if post_process is not None:
            for i, output_data in enumerate(output):
                output[i] = post_process(output_data)

        metric(output, target)
        cur_batch += 1

    return metric.compute()


def write_result_to_file(result):
    """Write model-related test results to a file for test report generation.

    Args:
        result (tuple, dict):
            tuple type: Will be converted to dict type before writing to the file.
                tuple's rules:
                    length: 6
                    Element per index position:
                        result[0]: runtime of Zhouyi Compass(rpc/sim), e.g. rpc
                        result[1]: framework and model name, e.g. tf_mobilenet_v1
                        result[2]: test metric, e.g. Cosine_Dis
                        result[3]: value
                        result[4]: threshold
                        result[5]: compare method(ge/le)
                            ge: value is greater than or equal to threshold
                            le: value is less than or equal to threshold

            dict type: will be written directly to the file.
    Raises:
        RuntimeError: if result is not a tuple or dict.
    """

    if os.environ.get("CPS_TVM_NIGHTLY_TEST", None) != "True":
        return

    if not isinstance(result, (tuple, dict)):
        raise RuntimeError("The type of 'result' must be a tuple or dict.")

    output_dir = CompassConfig.get().common["output_dir"]
    ir_path = os.path.join(output_dir, "partitioned_graph.txt")

    subgraph_count, op_count = get_subgraph_and_op_count(ir_path)
    npu_subgraph = {
        "NpuSubgraphNumber": {
            "value": subgraph_count,
            "threshold": 1,
            "cmp_method": "eq",
        }
    }
    other_op = {
        "OPNotInNpuNumber": {
            "value": op_count,
            "threshold": 0,
            "cmp_method": "eq",
        }
    }

    if isinstance(result, dict):
        result_dict = result
        model_name = list(result_dict.keys())[0]
        result_dict[model_name].update(npu_subgraph)
        result_dict[model_name].update(other_op)
    else:
        # (runtime, model name, test metric, value, threshold, cmp_method)
        assert len(result) == 6, "The length of 'result' must be 6."
        result_dict = {
            result[1]: {
                result[2]: {
                    "value": result[3],
                    "threshold": result[4],
                    "cmp_method": result[5],
                },
                "runtime": result[0],
            },
        }
        result_dict[result[1]].update(npu_subgraph)
        result_dict[result[1]].update(other_op)

    with open(NIGHTLY_TEST_TXT, "a") as f:
        f.write(json.dumps(result_dict) + "\n")


def calc_cos_distance(output_gt: dict, output_real: dict):
    """Calculate cosine distance."""

    def compare(out_gt, out_real):
        len_gt = len(out_gt)
        len_real = len(out_real)

        if len_gt == 0 or len_real == 0:
            return f"Length of outputs are 0, gt:{len_gt} vs real:{len_real}"

        if len_gt != len_real:
            return f"Length of outputs are not equal, gt:{len_gt} vs real:{len_real}"

        if out_gt.shape == [] and out_real.shape == []:
            return "Model output and Graph output are both empty"

        if np.any(out_gt < (-(2**30))) or np.any(np.isinf(out_gt)) or np.any(np.isnan(out_gt)):
            return "Model output maybe trustless, ignore this case"

        if np.array_equal(out_gt, out_real):
            return 1.0

        max_diff = max(abs(out_gt - out_real))
        if max_diff < 0.001:
            return 1.0

        cos_dis = compute_cos_distance(out_gt, out_real)
        if not is_number(cos_dis):
            return str(cos_dis)
        return cos_dis

    length_gt = len(output_gt)
    length_real = len(output_real)
    print(f"Number of output_gt: {length_gt}, Number of output_real: {length_real}")

    if length_gt != length_real:
        return f"Length of outputs are not equal, gt:{length_gt} vs real:{length_real}"

    if length_gt == 1:
        return compare(
            list(output_gt.values())[0].flatten(), list(output_real.values())[0].flatten()
        )

    if not all(key in output_real.keys() for key in output_gt.keys()):
        return f"Different output names, gt:{output_gt.keys()} vs real:{output_real.keys()}"

    sim = []
    for key in output_gt.keys():
        result = compare(output_gt[key].flatten(), output_real[key].flatten())
        if not is_number(result):
            return str(result)
        sim.append(float(result))

    return np.mean(sim)


def _get_quant_file_from_compass(compass_output_path):
    """Get quant IR and weight file from the compass_output* dir."""
    params = f"ls {compass_output_path}"
    with os.popen(params, "r") as p:
        subgraph_path_list = p.readlines()
    if len(subgraph_path_list) > 1:
        raise RuntimeError("Compass GT with multiple subgraphs is not supported yet.")

    rly_func_name = subgraph_path_list[0].strip()
    cfg = CompassFunctionConfig(rly_func_name)
    quant_nn_model_ir, quant_nn_model_bin = cfg.quant_compass_ir_path

    if not os.path.isfile(quant_nn_model_ir):
        raise FileNotFoundError(f"{quant_nn_model_ir} is not exists.")

    if not os.path.isfile(quant_nn_model_bin):
        raise FileNotFoundError(f"{quant_nn_model_bin} is not exists.")

    return quant_nn_model_ir, quant_nn_model_bin


def _get_scales_and_dtype(quant_info_input: list, quant_info_output: list):
    assert isinstance(quant_info_input, list), "The type of quant_info_input must be list."
    assert isinstance(quant_info_output, list), "The type of quant_info_output must be list."

    in_scale_list = []
    in_dtype_list = []
    out_scale_list = []
    out_dtype_list = []

    for quant_info in quant_info_input:
        assert isinstance(quant_info, dict), "The type of quant_info must be dict."
        assert "layer_top_scale" in quant_info.keys(), "Not found layer_top_scale."
        assert "layer_top_type" in quant_info.keys(), "Not found layer_top_type."
        in_scale_list.append(quant_info["layer_top_scale"])
        in_dtype_list.append(quant_info["layer_top_type"])

    for quant_info in quant_info_output:
        assert isinstance(quant_info, dict), "The type of quant_info must be dict."
        assert "layer_top_scale" in quant_info.keys(), "Not found layer_top_scale."
        assert "layer_top_type" in quant_info.keys(), "Not found layer_top_type."
        out_scale_list.append(quant_info["layer_top_scale"])
        out_dtype_list.append(quant_info["layer_top_type"])

    return in_scale_list, in_dtype_list, out_scale_list, out_dtype_list


def _prod(arr):
    return functools.reduce(operator.mul, arr, 1)


def _create_quant_info_list(layers, param_names):
    ret = [None] * len(param_names)

    # Each layer represents a block of information containing input or output names
    for layer in layers:
        for line in layer.split():
            key, value = (x.strip() for x in line.split("="))
            if key == "layer_top":
                names = [x.strip() for x in value.strip("[]").split(",")]
            elif key == "layer_top_shape":
                elem_cnts = [_prod(x) for x in json.loads(value)]
            elif key == "layer_top_type":
                dtypes = [x.strip() for x in value.strip("[]").split(",")]
            elif key == "layer_top_scale":
                scales = [float(x.strip()) for x in value.strip("[]").split(",")]

        for i, name in enumerate(names):
            if name not in param_names:
                continue
            ret[param_names.index(name)] = {
                "layer_top": names[i],
                "layer_top_shape": elem_cnts[i],
                "layer_top_type": dtypes[i],
                "layer_top_scale": scales[i],
            }

    return ret


def _parse_in_out_param_quant_info(quant_compass_ir_txt_path):
    with open(quant_compass_ir_txt_path) as f:
        q_cir = f.read()
    # Get the name list of input and output parameters.
    matches = re.search(r"input_tensors\s*=\s*\[(.+)\]", q_cir, re.MULTILINE)
    assert matches and len(matches.groups()) == 1
    in_param_names = [x.strip() for x in matches.group(1).split(",")]
    matches = re.search(r"output_tensors\s*=\s*\[(.+)\]", q_cir, re.MULTILINE)
    assert matches and len(matches.groups()) == 1
    out_param_names = [x.strip() for x in matches.group(1).split(",")]

    # Get the layer list of producing input and output parameters.
    in_param_layers = []
    out_param_layers = []
    for layer in q_cir.split("layer_type"):
        prefix = r"layer_top\s*=\s*\[.*("
        suffix = r").*\]"
        if re.search(f"{prefix}{'|'.join(in_param_names)}{suffix}", layer, re.MULTILINE):
            in_param_layers.append(layer)
        if re.search(f"{prefix}{'|'.join(out_param_names)}{suffix}", layer, re.MULTILINE):
            out_param_layers.append(layer)

    # Create object to store quantization information of each input and output
    # parameters.
    return (
        _create_quant_info_list(in_param_layers, in_param_names),
        _create_quant_info_list(out_param_layers, out_param_names),
    )


def _get_dtype_of_np(data_type: str):
    data_type = data_type.lower()

    dtype_mapping = {
        "int8": np.int8,
        "int16": np.int16,
        "int32": np.int32,
        "int64": np.int64,
        "uint8": np.uint8,
        "uint16": np.uint16,
        "uint32": np.uint32,
        "uint64": np.uint64,
    }

    if data_type not in dtype_mapping.keys():
        raise RuntimeError(f"Not supported data type: {data_type}")
    return dtype_mapping[data_type]


def compare_with_gt(input_data, tvm_output):
    """Compare the results of Compass TVM and that of Compass Gt."""

    input_data = convert_to_list(input_data)

    compass_output_path = CompassConfig.get().common["output_dir"]

    # 1, Get scales and data type of input and output from quant IR
    quant_ir, weight_file = _get_quant_file_from_compass(compass_output_path)
    quant_info_in_list, quant_info_out_list = _parse_in_out_param_quant_info(quant_ir)
    assert len(quant_info_in_list) == len(
        input_data
    ), "The length of the input data does not match the length of the input node in the quant IR."
    assert len(quant_info_out_list) == len(
        tvm_output
    ), "The length of the output data does not match the length of the output node in the quant IR."
    in_scale_list, in_dtype_list, out_scale_list, out_dtype_list = _get_scales_and_dtype(
        quant_info_in_list, quant_info_out_list
    )

    # 2, Create data path for Compass GT
    gt_data_path = os.path.join(compass_output_path, "gt_data")
    if not os.path.exists(gt_data_path):
        os.makedirs(gt_data_path)

    # 3, Quantify the input data and save it to bin file
    gt_input_list = []
    for i, in_data_float in enumerate(input_data):
        in_data_float = in_data_float * in_scale_list[i]
        in_data_int = in_data_float.astype(_get_dtype_of_np(in_dtype_list[i]))
        gt_input_file = f"{gt_data_path}/input_data_int_{i}.bin"
        in_data_int.tofile(gt_input_file)
        gt_input_list.append(gt_input_file)
    gt_input_path = ",".join(gt_input_list)

    # 4, Quantify the output data of Compass TVM
    tvm_output_list = []
    for i, out_data in enumerate(tvm_output):
        out_data_float = out_data.numpy()
        out_data_float = out_data_float * out_scale_list[i]
        tvm_output_int = out_data_float.astype(np.int32)
        tvm_output_list.append(tvm_output_int)

    # 5, Run Compass GT and get output
    target = CompassConfig.get().gbuilder["target"]
    print("Run GtGenerator.")
    run_gt_command = f"aipugt {quant_ir} -w {weight_file}\
            -i {gt_input_path} -o {gt_data_path} --id 'ir' --target {target}"
    os.system(run_gt_command)

    # 6, Compare results
    gt_output_list = []
    for f_name in os.listdir(gt_data_path):
        if not f_name.startswith("input_data"):
            gt_output = os.path.join(gt_data_path, f_name)
            gt_output_list.append(gt_output)

    for i, data in enumerate(gt_output_list):
        gt_output_data = np.fromfile(data, _get_dtype_of_np(out_dtype_list[i]))
        gt_length = len(gt_output_data.flatten())
        tvm_length = len(tvm_output_list[i].flatten())
        assert (
            gt_length == tvm_length
        ), f"Length of outputs are not equal: {gt_length} vs {tvm_length}"
        print(compute_cos_distance(gt_output_data, tvm_output_list[i]))


def get_tvm_output(cfg_file, input_data):
    """Simple wrapper of Compass TVM Forward."""
    compass = Compass(cfg_file)
    compass.parse()
    compass.optimize()
    compass.partition()
    is_partition_all_to_compass(compass.ir_mod)
    compass.collect_calibration_data()
    deployable = compass.build()
    execution_engine = deployable.create_execution_engine()
    output = execution_engine.run(*input_data)
    return output


def is_partition_all_to_compass(ir_mod):
    """Whether the whole graph is partitioned to Compass NPU graph or not."""

    def _analysis_no_call(t):
        if isinstance(t, relax.Call):
            assert str(t).startswith(
                "tvm_compass_"
            ), "Some operators aren't partitioned to Compass NPU subgraph."

    relax.analysis.post_order_visit(ir_mod["main"], _analysis_no_call)


def compare_relax_result(
    mod, *inputs, engine, compare=None, cfg=None, compare_threshold=0.99, golden=None
):
    """Compare relax output with user defined forward engine"""
    assert engine in ("opt_float", "opt_int", "gt")
    cfg_str = f"""
    [Common]
    forward_engine = {engine}

    [Parser]
    model_type = relax
    model_name = relax_test
    """
    if cfg:
        cfg_str = cfg

    compass = Compass(cfg_str)
    compass.ir_mod = mod
    compass.optimize()
    compass.partition()
    is_partition_all_to_compass(compass.ir_mod)
    if cfg:
        compass.collect_calibration_data()
    deployable = compass.build()
    execution_engine = deployable.create_execution_engine()

    outputs = execution_engine.run(*inputs)
    if golden is None:
        golden = relax.VirtualMachine(relax.build(mod, "llvm"), tvm.cpu())["main"](*inputs)

    if compare is None:
        if isinstance(golden, tvm.runtime.ndarray.NDArray):
            golden = [golden]

        for out, gold in zip(outputs, golden):
            cos_distance = compute_cos_distance(out.numpy(), gold.numpy())
            msg = f"Compare Failed, Real {cos_distance} vs Golden {compare_threshold}"
            assert cos_distance >= compare_threshold, msg
    else:
        compare(outputs, golden)


def compare_relax_opt_float_result(
    mod, *inputs, compare=None, cfg=None, compare_threshold=0.99, golden=None
):
    """Compare relax output with opt float forward"""
    compare_relax_result(
        mod,
        *inputs,
        engine="opt_float",
        compare=compare,
        cfg=cfg,
        compare_threshold=compare_threshold,
        golden=golden,
    )


def channel_manhattan_distance(data1, data2, op_sum, error_sum, channel, shape, ret_pixel):
    """Calculate the channel manhattan distance."""
    data1 = data1.reshape(tuple(shape))
    data2 = data2.reshape(tuple(shape))
    data_1 = data1.astype(float)
    data_2 = data2.astype(float)

    new_data = np.array(data_2[:, :, :, channel]).flatten()
    if ret_pixel:
        number = new_data.size
        print("Use pixel number for cals!")
    else:
        number = new_data[new_data != 0].size

    op = np.sum(np.abs(data_1[:, :, :, channel] - data_2[:, :, :, channel]))
    error = op / number

    op_sum += op
    error_sum += error
    return op, error, op_sum, error_sum


@timer
def calc_l1_norm_with_golden(
    engine: ExecutionEngine,
    input_dtypes: List[str],
    runtime: str,
    l1_norm_thresholds: List[float],
    max_batch_l1_norm: int = None,
    cfg_path: str = None,
):
    """Calculate L1 norm without run original framework."""
    cfg_parser = CompassConfig.get().parser
    model_type = cfg_parser["model_type"]
    model_name = cfg_parser["model_name"]
    model_name = f"{model_type}_{model_name}"

    # input names
    input_names = cfg_parser["input"]
    input_names = input_names.split(",")
    input_names = list(map(lambda s: s.strip(), input_names))

    # input shapes
    input_shapes_str = cfg_parser["input_shape"]
    if len(input_names) == 1:
        input_shapes = [json.loads(input_shapes_str)]
    else:
        input_shapes_str = input_shapes_str.replace(" ", "")
        input_shapes_str = input_shapes_str.replace("],[", "|")
        input_shapes_str = input_shapes_str.replace("]", "")
        input_shapes_str = input_shapes_str.replace("[", "")
        input_shapes = input_shapes_str.split("|")
        input_shapes = list(map(lambda s: s.split(","), input_shapes))
        input_shapes = [list(map(int, l)) for l in input_shapes]

    assert len(input_shapes) == len(input_names) == len(input_dtypes), (
        f"Error length for input elements: num_shapes({len(input_shapes)}),"
        " num_names({len(input_names)}), num_dtypes({len(input_dtypes)})"
    )

    # Common directories
    is_company_model_zoo = "CPS_TVM_DEV_AIB_PATH" in os.environ
    if is_company_model_zoo:
        model_data_dir = (
            f"{os.environ['CPS_TVM_DEV_AIB_PATH']}/{cfg_parser['model_name']}/tflite/1_15/config/"
        )
        in_img_dir = f"{model_data_dir}/input"
        out_golden_img_dir = f"{model_data_dir}/golden_output"

        # Max batch for compute L1 norm
        num_batch = len(os.listdir(in_img_dir)) // len(input_shapes)
        max_batch_l1_norm = num_batch if max_batch_l1_norm is None else max_batch_l1_norm
        if num_batch > max_batch_l1_norm:
            print(
                f"[WARN] input directory has {num_batch} inputs,"
                f" use max_batch_l1_norm({max_batch_l1_norm}) instead!"
            )
            assert max_batch_l1_norm > 0, (
                f"max_batch_l1_norm({max_batch_l1_norm})" " should be a non-negative number."
            )
            num_batch = max_batch_l1_norm
    else:
        model_dir = os.path.dirname(cfg_parser["input_model"])
        in_img_dir = model_dir
        out_golden_img_dir = None
        num_batch = 1

    # Run execution engine
    tvm_output_list = []
    golden_output_list = []
    for batch_idx in range(1, num_batch + 1):
        # input
        inputs = [None] * len(input_names)
        for iidx, inp_name in enumerate(input_names):
            if is_company_model_zoo:
                if len(input_names) == 1:
                    inp_path = f"{in_img_dir}/input_{batch_idx}.bin"
                else:
                    inp_path = f"{in_img_dir}/input{batch_idx}_{inp_name}.bin"
            else:
                if len(input_names) == 1:
                    inp_path = f"{in_img_dir}/input.bin"
                else:
                    inp_path = f"{in_img_dir}/input{batch_idx}_{inp_name}.bin"
            inp = np.fromfile(inp_path, input_dtypes[iidx])
            shape = input_shapes[iidx]
            inputs[iidx] = np.reshape(inp, shape)

        outputs = engine.run(*inputs)

        if is_company_model_zoo:
            tvm_output_dict = dict()
            golden_output_dict = dict()
            for oidx, out in enumerate(outputs):
                out_key = f"out_{oidx}"
                # output from Compass TVM.
                tvm_output_dict.update({out_key: out.numpy().flatten()})

                # output from golden truth
                suffix = str(oidx) if oidx != 0 else ""
                golden_file = f"{out_golden_img_dir}/output_{batch_idx}.bin{suffix}"
                if model_name.split("_", 1)[1] == "mobilebert_quant":
                    suffix = "start" if oidx == 0 else "end"
                    golden_file = f"{out_golden_img_dir}/output{batch_idx}_{suffix}_logits.bin"
                golden_output = np.fromfile(golden_file, out.dtype)
                golden_output_dict.update({out_key: golden_output})
        else:
            assert (
                cfg_parser["model_type"] == "tflite"
            ), f"Unsupported model type:{cfg_parser['model_type']}"
            model_inst = TFLiteModel(cfg_path)
            tvm_output_dict, golden_output_dict = get_output_dict(model_inst, inputs, outputs)

        tvm_output_list.append(tvm_output_dict)
        golden_output_list.append(golden_output_dict)

    calc_l1_norm(tvm_output_list, golden_output_list, model_name, l1_norm_thresholds, runtime)


def calc_l1_norm(
    data1: List[dict], data2: List[dict], model_name: str, threshold: List[float], runtime: str
):
    """
    Calculate the L1 Norm.
    :param data1: the first data, usually the output data of compass
    :param data2: the second data, usually the output data of original frame
    :param model_name: model name, e.g. tflite_vgg_quant
    :param threshold: threshold, e.g. [0.46]
    :param runtime: runtime, e.g. rpc/sim
    :return: None
    """
    assert len(data1) == len(data2), "The length of the two data must be same."

    framework, model_name = model_name.split("_", 1)
    assert (
        model_name in AI_BENCHMARK_PERCHANNEL or model_name in AI_BENCHMARK_NOZERO
    ), f"Please add output shape of {model_name} to the AI BENCHMARK dict."
    if model_name in AI_BENCHMARK_PERCHANNEL:
        output_shape = AI_BENCHMARK_PERCHANNEL[model_name]
        ret_pixel = True
    else:
        output_shape = AI_BENCHMARK_NOZERO[model_name]
        ret_pixel = False

    input_num = len(data1)
    op_sum = [0] * output_shape[3]
    norm_sum = [0] * output_shape[3]

    for i in range(0, input_num):
        data1_dict, data2_dict = data1[i], data2[i]
        if model_name == "mobilebert_quant":
            assert len(data1_dict) == len(data2_dict) == 2
            data1_arr = np.zeros(shape=output_shape[1:3])
            data2_arr = np.zeros(shape=output_shape[1:3])
            data1_arr[1] = data1_dict.popitem()[1]
            data1_arr[0] = data1_dict.popitem()[1]
            data2_arr[1] = data2_dict.popitem()[1]
            data2_arr[0] = data2_dict.popitem()[1]
        else:
            # Here, we align with Top test:
            # The number of L1 norm is determined by the channel of frist output.
            # For example, the output shape of deeplab_v3_quant is [1,256,256,19] owning 19 numbers.
            # When some model owns multiple outputs such as yolo_v4_tiny or mobilebert both owning
            # 2 outputs, the Top test only load first one and compute L1 norm (detailed see
            # function `compute_L1_norm` in `AIPU_common/top_test/end_to_end_test/
            # e2e_test_tool.py`).
            # The number of L1 norm of mobilebert is 1. However, yolo_v4_tiny has 2,
            # but in fact the second value is 0, which is meaningless.
            # Thus, here skip non-first output when compute L1 norm.
            assert (
                len(data1_dict) > 0 and len(data2_dict) > 0
            ), "The length of the two data must be positive."
            # Here only keep first output.
            while len(data1_dict) > 1:
                data1_dict.popitem()
                data2_dict.popitem()
            data1_arr = data1_dict.popitem()[1]
            data2_arr = data2_dict.popitem()[1]

        for j in range(0, output_shape[3]):
            op, error, op_sum[j], norm_sum[j] = channel_manhattan_distance(
                data1_arr, data2_arr, op_sum[j], norm_sum[j], j, output_shape, ret_pixel
            )
            print("Channel:%s, " % str(j))
            print("Manhattan distance:%.3f, " % op)
            print("L1 norm:%.3f \n" % error)

    assert len(threshold) == output_shape[3]
    l1_norms = []
    for i in range(0, output_shape[3]):
        print("[Sum] Channel:%s, " % str(i))
        print("Total Manhattan distance:%.3f, " % op_sum[i])
        print("Total norm sum:%.3f, " % (float(norm_sum[i])))
        l1_norm = round(float(norm_sum[i]) / float(input_num), 2)
        l1_norms.append(l1_norm)
        print("Total L1 norm:%.2f \n" % (l1_norm))

        write_result_to_file(
            (
                runtime,
                "_".join([framework, model_name]),
                f"L1_norm_channel_{i}",
                l1_norm,
                threshold[i],
                "le",
            )
        )

    for i in range(0, output_shape[3]):
        l1, thresh = float(l1_norms[i]), float(threshold[i])
        print(f"L1 Norm Channel {i} of {model_name}: gt={thresh} vs real={l1}")
        if thresh > 10 or l1 <= thresh:
            continue
        diff = l1 - thresh
        assert_info = (
            f"L1 Norm Channel {i} of {model_name} is FAILED, "
            f"gt={thresh} vs real={l1}, the diff is greater than 5."
        )
        assert diff <= 5, assert_info


def get_output_dict(model_inst, input_data, tvm_output):
    """Get output dict."""
    # Check if model instance is valid
    if not isinstance(model_inst, (TFModel, TFLiteModel, ONNXModel, TorchModel, RelaxModel)):
        raise RuntimeError("The model instance is not valid.")

    print("Run original framework forward.")
    model_out_dict = model_inst.run(input_data)

    tvm_out_dict = tvm_output
    if not isinstance(tvm_output, dict):
        tvm_out_dict = {}
        for i, out_name in enumerate(model_inst.output_tensor_names):
            _out = tvm_output[i]
            tvm_out_dict.update({out_name: _out.numpy().flatten()})

    return tvm_out_dict, model_out_dict


def get_test_result(model, input_data, tvm_output, threshold=0.90, runtime="sim"):
    """Compare the results of Compass TVM and the original framework forward."""
    tvm_out_dict, model_out_dict = get_output_dict(model, input_data, tvm_output)
    cos_distance = calc_cos_distance(model_out_dict, tvm_out_dict)
    print(f"The Cosine Distance of {model.model_name}: {cos_distance}")
    write_result_to_file(
        (
            runtime,
            "_".join([model.model_type, model.model_name]),
            "Cosine_Dis",
            cos_distance,
            threshold,
            "ge",
        )
    )
    assert is_number(cos_distance), cos_distance
    assert (
        float(cos_distance) >= threshold
    ), f"Cosine Distance: {model.model_name} compare FAILED, gt:{threshold} vs real:{cos_distance}"


def get_topk_result(
    execution_engine,
    model_name,
    img_dataset,
    img_number,
    threshold=None,
    im_height=224,
    im_width=224,
    preprocess_mode="tf_vgg",
    runtime="sim",
):
    """
    Calculate the Top-1 and Top-5 Accuracy.

    Parameters:
            execution_engine: execution engine, including compass tvm and source framework forward
            model_name: model name, must be in this format: framework_model
            img_dataset: The image dataset of Classification Model
            img_number: The number of pictures
            threshold (Optional, list/tuple): threshold for topk, default to 0
            im_height (Optional): The height of input data, default to 224
            im_width (Optional): The width of intput data, deafult to 224
            preprocess_mode (Optional): preprocess mode, default to tf_vgg
            runtime (Optional): runtime of Compass, defaults to simulator
            (For compatibility with existing test program)

    Returns:
            None
    """
    if not any([model_name.startswith(key) for key in FRAMEWORK_LIST]):
        raise RuntimeError(f"{model_name} is not valid, must be in format: framework_model")

    assert isinstance(img_dataset, ClassificationDataset), "Must be a Classification Dataset."

    # Threshold default to 0
    top1_thresh, top5_thresh = 0, 0
    if threshold is not None:
        assert (
            isinstance(threshold, (list, tuple)) and len(threshold) == 2
        ), "Threshold must be a list or tuple of length 2."
        top1_thresh, top5_thresh = threshold

    # Get the specified number of image sets
    img_list = img_dataset.get_img_list(img_number)
    # Init the true positive number
    positive_number_top1 = 0
    positive_number_top5 = 0
    # Iter through the image list to run forward
    for img in img_list:
        # Preprocess the input image
        img_path = img_dataset.get_img_path(img)
        image = get_imagenet_input(im_height, im_width, preprocess_mode, img_path)
        # Run
        outputs = execution_engine.run(image)
        # Postprocess the output data
        if isinstance(outputs, ir.Array):
            # output from Compass TVM
            predictions = outputs[0].numpy()[0]
        else:
            # output from source model framework
            if len(outputs) == 1:
                for _, v in outputs.items():
                    predictions = v
            else:
                raise RuntimeError(f"Expected 1 output vs {len(outputs)} output.")
        top1_idxes = np.argsort(predictions)[-1:]
        top5_idxes = np.argsort(predictions)[-5:][::-1]
        real_label = img_dataset.get_img_label(img)
        # Check if there is a matching label in the result list
        if real_label in top1_idxes.tolist():
            positive_number_top1 += 1
        if real_label in top5_idxes.tolist():
            positive_number_top5 += 1

    # Calculate the top-k accuracy
    top1_acc = positive_number_top1 / img_number
    top5_acc = positive_number_top5 / img_number
    print(f"---Validatoin on {img_number} images---")
    print(f"The Top-1 accuracy of {model_name}: {top1_acc}, threshold: {top1_thresh}")
    print(f"The Top-5 accuracy of {model_name}: {top5_acc}, threshold: {top5_thresh}")

    result_dict = {
        model_name: {
            "TOP-1": {
                "value": top1_acc,
                "threshold": top1_thresh,
            },
            "TOP-5": {
                "value": top5_acc,
                "threshold": top5_thresh,
            },
            "runtime": runtime,
        }
    }
    write_result_to_file(result_dict)
    assert (
        top1_acc >= top1_thresh
    ), f"TOP-1: {model_name} compare FAILED, gt:{top1_thresh} vs real:{top1_acc}"
    assert (
        top5_acc >= top5_thresh
    ), f"TOP-5: {model_name} compare FAILED, gt:{top5_thresh} vs real:{top5_acc}"
