# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2023-2024 Arm Technology (China) Co. Ltd.
import os
import random
import pytest
from PIL import Image, ImageDraw
import matplotlib.pyplot as plt
import tvm
from tvm import relay
from tvm.relay.backend.contrib.aipu_compass import AipuCompass
from tvm.relay.backend.contrib.aipu_compass import testing as aipu_testing
from tvm.relay.backend.contrib.aipu_compass.testing.gen_model_inputs import CoCo
from tvm.relay.backend.contrib.aipu_compass.testing.data_processing import draw_box
from tvm.relay.op.contrib.aipu_compass import op as aipu_op
from tvm.relay.backend.contrib.aipu_compass.transform.find_aipu_postprocess import (
    find_main_dependency_stack,
)


@tvm.ir.transform.module_pass(opt_level=0)
class InsertOps:
    """
    Insert decodebox and nms for this model.
    """

    def insert_ops_fn(self, all_nodes):
        scores, boxes = None, None
        for node in all_nodes:
            if not isinstance(node, relay.Call):
                continue
            if relay.ty.is_dynamic(node.checked_type):
                continue
            if node.op == relay.op.get("sigmoid"):
                shape = [int(i) for i in node.checked_type.shape]
                if shape == [1, 1917, 91]:
                    scores = node
            if node.op == relay.op.get("squeeze"):
                shape = [int(i) for i in node.checked_type.shape]
                if shape == [1, 1917, 4]:
                    boxes = node

        assert scores and boxes
        feature_map = [[19, 19], [10, 10], [5, 5], [3, 3], [2, 2], [1, 1]]
        decode_box = aipu_op.decode_box(scores, boxes, feature_map, score_threshold=0.55)
        nms = aipu_op.nms(decode_box[0], decode_box[1], decode_box[2], decode_box[3])
        return relay.Tuple(list(nms) + list(decode_box))

    def transform_module(self, mod, ctx):  # pylint: disable=unused-argument
        """
        Function to transform module
        """
        update_mod = mod
        all_nodes = find_main_dependency_stack(update_mod["main"].body)
        new_call = self.insert_ops_fn(all_nodes)
        new_func = relay.Function(update_mod["main"].params, new_call)
        for gvar in update_mod.functions:
            if gvar.name_hint == "main":
                update_mod.update_func(gvar, new_func)
        update_mod = relay.transform.InferType()(update_mod)

        return update_mod


def map_ground_truth_box(bbox):
    bbox_x, bbox_y, bbox_w, bbox_h = bbox
    return [bbox_y, bbox_x, bbox_y + bbox_h, bbox_x + bbox_w]


def post_process(outputs, img_height, img_width):
    outputs = [out.numpy()[0] for out in outputs]
    boxes = outputs[0]
    num_per_class = outputs[1].astype("int32")
    scores = outputs[2]
    class_num = outputs[6].astype("int32")
    classes = outputs[8].astype("int32")
    out_classes = []
    total_num = 0
    for i in range(class_num[0]):
        out_classes += [classes[i] + 1] * num_per_class[i]
        total_num += num_per_class[i]

    out_boxes = boxes[:total_num]
    out_boxes[:, 0:1] *= img_height  # ymin*height
    out_boxes[:, 1:2] *= img_width  # xmin*width
    out_boxes[:, 2:3] *= img_height  # ymax*height
    out_boxes[:, 3:4] *= img_width  # xmax*width
    out_boxes = out_boxes.astype("int32")

    out_scores = scores[:total_num]

    return out_boxes, out_scores, out_classes


def compass_compile(compass, target):
    compass.parse()
    compass.optimize()
    compass.ir_mod = InsertOps()(compass.ir_mod)  # pylint: disable=not-callable
    compass.partition()
    compass.collect_calibration_data()
    return compass.build(target=target)


def run_ssd_mobilenet(
    model, runtime="sim", visualize=False, calc_mAP=False, imgs_number=1, map_thres=0
):
    cfg = f"{aipu_testing.DATA_DIR}/onnx_{model}.cfg"
    # 1. Create AIPU Compass instance and set configurations.
    compass = AipuCompass(cfg)

    # 2. Compile the nn model.
    target = "llvm -mtriple=aarch64-linux-gnu" if runtime == "rpc" else "llvm"
    deployable = compass_compile(compass, target)

    # 3. Create execution engine.
    rpc_sess, device_compiler = None, None
    if runtime == "rpc":
        rpc_sess = aipu_testing.get_rpc_session()
        device_compiler = aipu_testing.DEVICE_COMPILER
        assert device_compiler is not None, "need to set device cross compiler path."
    ee = deployable.create_execution_engine(rpc_sess, cc=device_compiler)

    # random select an image
    coco = CoCo()
    img_ids = coco.get_img_ids()
    random.seed(5)
    img_id = random.sample(img_ids, 1)
    img = coco.load_imgs(img_id)[0]
    img_path = coco.get_img_path(img["file_name"])
    annotation = coco.load_anns(coco.get_ann_ids(img_ids=img_id))
    image = aipu_testing.ssd_mobilenet_preprocess(img_path, im_height=300, im_width=300)

    # 4. Run the nn model.
    outputs = ee.run(image)
    out_boxes, out_scores, out_classes = post_process(outputs, img["height"], img["width"])

    print(f"total detect {len(out_classes)} objects:")
    for box, cat in zip(out_boxes, out_classes):
        if cat in coco.cats.keys():
            label = coco.load_cats(int(cat))[0]["name"]
            print(
                f"{label} with bounding box top:{box[0]}, left:{box[1]}, bottom:{box[2]}, right:{box[3]}"
            )

    if visualize:
        ground_truth_boxes = [ann["bbox"] for ann in annotation]
        ground_truth_classes = [ann["category_id"] for ann in annotation]
        img = Image.open(img_path)
        draw = ImageDraw.Draw(img)
        for box, cat in zip(out_boxes, out_classes):
            if cat in coco.cats.keys():
                label = coco.load_cats(int(cat))[0]["name"]
                draw_box(draw, box, thickness=2, color="yellow", label=label)

        for box, cat in zip(ground_truth_boxes, ground_truth_classes):
            mapped_box = map_ground_truth_box(box)
            label = coco.load_cats(cat)[0]["name"]
            draw_box(draw, mapped_box, thickness=2, color="red", label=label)

        plt.figure(figsize=(80, 40))
        plt.axis("off")
        plt.imshow(img)
        plt.show()

    if calc_mAP:
        predict = {}
        img_ids = random.sample(img_ids, imgs_number)
        for img_id in img_ids:
            img = coco.load_imgs(img_id)[0]
            img_path = coco.get_img_path(img["file_name"])
            outputs = ee.run(
                aipu_testing.ssd_mobilenet_preprocess(img_path, im_height=300, im_width=300)
            )

            out_boxes, out_scores, out_classes = post_process(outputs, img["height"], img["width"])

            out_scores = [float(score) for score in out_scores]
            out_boxes = [[float(i) for i in box] for box in out_boxes]
            predict[img_id] = {
                "scores": out_scores,
                "boxes": out_boxes,
                "categroies": out_classes,
            }
        mean_ap = aipu_testing.calc_mean_ap(
            coco, predict, model, "onnx", map_thres=map_thres, runtime=runtime
        )
        print(f"On MSCoCo 2017 validation dataset with IoU 0.5, the mAP is {mean_ap}")


@pytest.mark.X2_1204
@pytest.mark.parametrize("runtime", ("rpc", "sim"))
@aipu_testing.clear_traceback
def test_mobilenet_v2_ssd(runtime):
    if os.environ.get("AIPU_TVM_NIGHTLY_TEST", None) == "True":
        run_ssd_mobilenet("mobilenet_v2_ssd", runtime, False, True, 10, map_thres=0.42)
    else:
        run_ssd_mobilenet("mobilenet_v2_ssd", runtime)


if __name__ == "__main__":
    test_mobilenet_v2_ssd("sim")
