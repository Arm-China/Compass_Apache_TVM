# This file is CONFIDENTIAL and created by Arm Technology (China) Co., Ltd.
# See the copyright file distributed with this work for additional information
# regarding copyright ownership.
import os
import random
import pytest
from PIL import Image, ImageDraw
import matplotlib.pyplot as plt
from tvm.relay.backend.contrib.aipu_compass import AipuCompass
from tvm.relay.backend.contrib.aipu_compass import testing as aipu_testing
from tvm.relay.backend.contrib.aipu_compass.testing.gen_model_inputs import VOC
from tvm.relay.backend.contrib.aipu_compass.testing.data_processing import (
    PostprocessYOLOv2 as PostProcessor,
    draw_box,
)


def map_ground_truth_box(bbox):
    xmin, ymin, xmax, ymax = bbox
    return [ymin, xmin, ymax, xmax]


def _show_result(boxes, classes, voc, img_id):
    print(f"Total detect {len(classes)} objects:")
    for box, cat in zip(boxes, classes):
        label = voc.load_cats(cat)[0]
        print(
            f"{label} with bounding box top:{box[0]}, left:{box[1]}, bottom:{box[2]}, "
            f"right:{box[3]}"
        )

    if os.getenv("SKIP_VISUALIZE") is not None:
        return
    img = Image.open(voc.load_imgs(img_id)[0]["file_path"])
    draw = ImageDraw.Draw(img)

    for box, cat in zip(boxes, classes):
        label = voc.load_cats(cat)[0]
        draw_box(draw, box, thickness=2, color="yellow", label=label)

    anns = voc.load_anns(img_id)
    ground_truth_boxes = [ann["bbox"] for ann in anns]
    ground_truth_classes = [ann["category_id"] for ann in anns]
    for box, cat in zip(ground_truth_boxes, ground_truth_classes):
        mapped_box = map_ground_truth_box(box)
        label = voc.load_cats(cat)[0]
        draw_box(draw, mapped_box, thickness=2, color="green", label=label)

    plt.figure(figsize=(80, 40))
    plt.axis("off")
    plt.imshow(img)
    plt.show()


def run_yolo_v2_416(runtime, imgs_number=10):
    cfg = f"{aipu_testing.DATA_DIR}/onnx_yolo_v2_416.cfg"
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

    # 4. Run and check result.
    # a. Prepare testing dataset.
    voc = VOC()
    # Randomly select some samples.
    random.seed(1)
    all_img_ids = voc.get_img_ids()
    random.sample(all_img_ids, 1)  # Skip the first several unfriendly samples.
    img_ids = random.sample(all_img_ids, imgs_number)
    anchors = [
        [1.3221, 1.73145],
        [3.19275, 4.00944],
        [5.05587, 8.09892],
        [9.47112, 4.84053],
        [11.2364, 10.0071],
    ]

    # b. Check cosine distance with original framework on simulator.
    if runtime == "simulator":
        img_id = all_img_ids[0]
        img = voc.load_imgs(img_id)[0]
        print(f"Input image: {img}")
        preprocessed_image = aipu_testing.yolo_v2_preprocess(
            img["file_path"], im_height=416, im_width=416
        )
        outputs = ee.run(preprocessed_image)

        boxes, scores, classes = PostProcessor(
            image_shape=(img["height"], img["width"]), anchors=anchors, num_categories=20
        ).process(outputs)
        _show_result(boxes, classes, voc, img_id)

        onnx_model = aipu_testing.ONNXModel(cfg)
        aipu_testing.get_test_result(onnx_model, preprocessed_image, outputs, runtime=runtime)
        return

    # c. Check metric.
    predict = {}
    for img_id in img_ids:
        img = voc.load_imgs(img_id)[0]
        outputs = ee.run(
            aipu_testing.yolo_v2_preprocess(img["file_path"], im_height=416, im_width=416)
        )

        boxes, scores, classes = PostProcessor(
            image_shape=(img["height"], img["width"]), anchors=anchors, num_categories=20
        ).process(outputs)

        _show_result(boxes, classes, voc, img_id)

        predict[img_id] = {
            "scores": scores,
            "boxes": boxes,
            "categroies": classes,
        }
    mean_ap = aipu_testing.calc_mean_ap(
        voc, predict, "yolo_v2_416", "onnx", map_thres=0.80, runtime=runtime
    )
    print(f"On VOC2007 test dataset with IoU 0.5, the mAP is {mean_ap}")


@pytest.mark.X2_1204
@pytest.mark.parametrize("runtime", ["rpc", "simulator"])
@aipu_testing.clear_traceback
def test_yolo_v2_416(runtime):
    run_yolo_v2_416(runtime)


if __name__ == "__main__":
    run_yolo_v2_416("simulator")
