# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2023 Arm Technology (China) Co. Ltd.
"""Generate the input data for Model test."""
# pylint: disable=invalid-name
import os
import random
from collections import defaultdict
import json
import time
import itertools
import abc
import xml.etree.ElementTree as ET
import numpy as np
from PIL import Image
import cv2
from tvm.contrib import download


def get_real_image(im_height=None, im_width=None):
    """Get one image with specified size."""
    img_url = "https://github.com/dmlc/mxnet.js/blob/main/data/cat.png?raw=true"
    img_path = download.download_testdata(img_url, "cat.png", module="data")
    img = Image.open(img_path)
    height = img.height
    width = img.width
    if im_height is None and im_width is not None:
        scale = im_width / width
        im_height = int(scale * height)
    elif im_height is not None and im_width is None:
        scale = im_height / height
        im_width = int(scale * width)
    elif im_height is None and im_width is None:
        return img
    return img.resize((im_height, im_width))


def get_imagenet_input(im_height=224, im_width=224, preprocess_mode="tf_vgg", img_path=None):
    """Get one single image that preprocessed through the common method used by
    ImageNet.
    If img_path is None, then use the default image pre downloaded by get_real_image().
    """
    if img_path:
        # Image's layout in here may is (H, W, 3) or (H, W)
        resized_image = Image.open(img_path).resize((im_height, im_width))
    else:
        # Image's layout is "HWC", so here layout is (H, W, 3).
        resized_image = get_real_image(im_height, im_width)
    image_data = np.asarray(resized_image)
    if len(image_data.shape) == 2:
        # Add a dimension to get "HWC" layout.
        image_data = np.expand_dims(image_data, axis=(-1))
        # Convert GRAY to RGB by repeat the channel dimension.
        image_data = np.repeat(image_data, 3, -1)
    # Add a dimension to get "NHWC" layout.
    image_data = np.expand_dims(image_data, axis=0)

    if preprocess_mode == "none":
        return image_data

    image_data = image_data.astype("float32")
    # Preprocess image as below code of "TensorFlow-Slim".
    # https://github.com/tensorflow/models/tree/master/research/slim/preprocessing
    if preprocess_mode == "tf_vgg":
        image_data[:, :, :, 0] -= 123.68  # _R_MEAN
        image_data[:, :, :, 1] -= 116.78  # _G_MEAN
        image_data[:, :, :, 2] -= 103.94  # _B_MEAN
    elif preprocess_mode == "tf_inc":
        image_data /= 127.5
        image_data -= 1.0
    elif preprocess_mode == "tf_alex":
        image_data /= 255
    elif preprocess_mode == "caffe":
        image_data[:, :, :, 0] -= 123.68
        image_data[:, :, :, 1] -= 116.78
        image_data[:, :, :, 2] -= 103.94
        image_data *= 0.017
    elif preprocess_mode == "de_tr":
        image_data[:, :, :, 0] = (image_data[:, :, :, 0] - 123.675) / 58.395
        image_data[:, :, :, 1] = (image_data[:, :, :, 1] - 116.280) / 57.120
        image_data[:, :, :, 2] = (image_data[:, :, :, 2] - 103.530) / 57.375
    elif preprocess_mode == "paddle":
        # https://github.com/PaddlePaddle/PaddleClas/blob/4a1bdf585785401f6301aeb00c1d6ceb8f376c29/deploy/python/preprocess.py#L302
        scale = np.float32(1.0 / 255.0)
        image_data[:, :, :, 0] = (image_data[:, :, :, 0] * scale - 0.485) / 0.229
        image_data[:, :, :, 1] = (image_data[:, :, :, 1] * scale - 0.456) / 0.224
        image_data[:, :, :, 2] = (image_data[:, :, :, 2] * scale - 0.406) / 0.225
    return image_data


def letterbox_image(image, size):
    """resize image with unchanged aspect ratio using padding"""
    i_width, i_height = image.size
    width, height = size
    scale = min(width / i_width, height / i_height)
    new_width = int(i_width * scale)
    new_height = int(i_height * scale)

    image = image.resize((new_width, new_height), Image.BICUBIC)
    new_image = Image.new("RGB", size, (128, 128, 128))
    new_image.paste(image, ((width - new_width) // 2, (height - new_height) // 2))
    return new_image


def yolo_v3_preprocess(img, im_height=416, im_width=416):
    """
    Preprocess refer to
    https://github.com/onnx/models/tree/master/vision/object_detection_segmentation/yolov3
    """
    if isinstance(img, str):
        img = Image.open(img)

    boxed_image = letterbox_image(img, (im_width, im_height))
    image_data = np.array(boxed_image, dtype="float32")
    image_data /= 255.0
    image_data = np.transpose(image_data, [2, 0, 1])
    image_data = np.expand_dims(image_data, 0)
    return image_data


def yolo_v3_608_preprocess(image_file, im_height=608, im_width=608):
    """
    Preprocess refer to
    https://github.com/linghu8812/YOLOv3-TensorRT
    """

    image = Image.open(image_file)
    image = image.resize((im_height, im_width), resample=Image.BICUBIC)
    image = np.array(image, dtype="float32", order="C")
    image = np.expand_dims(image, axis=0)
    image /= 255.0
    image = np.transpose(image, [0, 3, 1, 2])

    return image


def yolo_v4_preprocess(img, im_height=416, im_width=416):
    """
    Preprocess refer to
    https://github.com/onnx/models/tree/main/vision/object_detection_segmentation/yolov4
    """
    assert os.path.isfile(img), f"{img} is not exists."
    image = cv2.imread(img)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    ih, iw = im_height, im_width
    h, w, _ = image.shape

    scale = min(iw / w, ih / h)
    nw, nh = int(scale * w), int(scale * h)
    image_resized = cv2.resize(image, (nw, nh))

    image_padded = np.full(shape=[ih, iw, 3], fill_value=128.0)
    dw, dh = (iw - nw) // 2, (ih - nh) // 2
    image_padded[dh : nh + dh, dw : nw + dw, :] = image_resized
    image_padded = image_padded / 255.0
    image_data = image_padded[np.newaxis, ...].astype(np.float32)

    return image_data


def yolo_v2_preprocess(image_file, im_height=416, im_width=416):
    """
    Preprocess refer to
    https://github.com/onnx/models/tree/main/vision/object_detection_segmentation/yolov2-coco
    """
    image = cv2.imread(image_file)
    # Convert to RGB format
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    # Resize to 416*416
    image_resized = cv2.resize(image_rgb, (im_width, im_height))
    # Normalize
    image_normalized = image_resized.astype(np.float32) / 255.0
    # Convert HWC to CHW
    image_normalized = np.transpose(image_normalized, [2, 0, 1])
    # Convert CHW to NCHW
    image_expanded = np.expand_dims(image_normalized, axis=0)
    return image_expanded


def ssd_mobilenet_preprocess(img, im_height=300, im_width=300):
    """
    Preprocess refer to
    https://github.com/tensorflow/models/blob/archive/research/object_detection/models/ssd_mobilenet_v2_feature_extractor.py
    """
    if isinstance(img, str):
        img = Image.open(img).resize([im_height, im_width])

    image_data = np.array(img, dtype="float32")
    image_data = (image_data * 2.0 / 255.0) - 1.0
    image_data = np.transpose(image_data, [2, 0, 1])
    image_data = np.expand_dims(image_data, 0)
    return image_data


def ssd_resnet_preprocess(img, im_height=300, im_width=300):
    """
    Preprocess refer to
    https://github.com/IntelAI/models/tree/master/models/object_detection/tensorflow/ssd-resnet34/inference/fp32/infer_detections.py
    """
    if isinstance(img, str):
        img = Image.open(img).resize([im_height, im_width])

    image_data = np.array(img, dtype="float32")
    image_data = image_data / 255.0
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    image_data = (image_data - mean) / std
    image_data = np.expand_dims(image_data, 0)

    return image_data


def laneaf_preprocess(image_file):
    """
    Preprocess refer to
    https://github.com/sel118/LaneAF/blob/main/datasets/culane.py
    """
    img = cv2.imread(image_file).astype("float32") / 255.0
    img = cv2.resize(img[14:, :, :], (1664, 576), interpolation=cv2.INTER_LINEAR)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_LINEAR)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    img = (img - mean) / std
    img = np.expand_dims(img, 0)

    return img


def get_imagenet_synset(class_cnt=1000):
    """Get class id to human readable string dictionary of ImageNet."""
    synset_name = f"imagenet{class_cnt}_clsid_to_human.txt"
    synset_url = (
        "https://gist.githubusercontent.com/Johnson9009/"
        "ca74433eda0117b2c9093df6deea2c7f/raw/"
        f"eb576a762df630812cf9fe9fbd8f79ebb79edbad/{synset_name}"
    )
    synset_path = download.download_testdata(synset_url, synset_name, module="data")
    with open(synset_path) as f:
        return eval(f.read())  # pylint: disable=eval-used


def _is_array_like(obj):
    return hasattr(obj, "__iter__") and hasattr(obj, "__len__")


class CoCo:
    """
    Helper class to get coco dataset info
    """

    def __init__(self, annotation_file=None):
        """
        Constructor of Microsoft coco helper class for reading and visualizing annotations.
        :param annotation_file (str): location of annotation file
        :param image_folder (str): location to the folder that hosts images.
        :return:
        """
        # load dataset
        self.dataset, self.anns, self.cats, self.imgs = dict(), dict(), dict(), dict()
        self.img_to_anns, self.cat_to_imgs = defaultdict(list), defaultdict(list)
        # Default initialized to validation dataset
        if annotation_file is None:
            annotation_file = (
                f"{os.environ['ZHOUYI_DATASET_HOME']}/MSCOCO/data/coco2017"
                "/annotations/instances_val2017.json"
            )
        print("loading annotations into memory...")
        tic = time.time()
        dataset = json.load(open(annotation_file, "r"))
        assert isinstance(dataset, dict), "annotation file format {} not supported".format(
            type(dataset)
        )
        print("Done (t={:0.2f}s)".format(time.time() - tic))
        self.dataset = dataset
        self.create_index()

    def create_index(self):
        """
        create index
        """
        print("creating index...")
        anns, cats, imgs = {}, {}, {}
        img_to_anns, cat_to_imgs = defaultdict(list), defaultdict(list)
        if "annotations" in self.dataset:
            for ann in self.dataset["annotations"]:
                img_to_anns[ann["image_id"]].append(ann)
                anns[ann["id"]] = ann

        if "images" in self.dataset:
            for img in self.dataset["images"]:
                imgs[img["id"]] = img

        if "categories" in self.dataset:
            for cat in self.dataset["categories"]:
                cats[cat["id"]] = cat

        if "annotations" in self.dataset and "categories" in self.dataset:
            for ann in self.dataset["annotations"]:
                cat_to_imgs[ann["category_id"]].append(ann["image_id"])

        print("index created!")
        # create class members
        self.anns = anns
        self.img_to_anns = img_to_anns
        self.cat_to_imgs = cat_to_imgs
        self.imgs = imgs
        self.cats = cats

    def info(self):
        """
        Print information about the annotation file.
        :return:
        """
        for key, value in self.dataset["info"].items():
            print("{}: {}".format(key, value))

    def get_ann_ids(self, img_ids=None, cat_ids=None, area_rng=None, iscrowd=False):
        """
        Get ann ids that satisfy given filter conditions. default skips that filter
        :param img_ids  (int array)     : get anns for given imgs
               cat_ids  (int array)     : get anns for given cats
               area_rng (float array)   : get anns for given area range (e.g. [0 inf])
               iscrowd (boolean)       : get anns for given crowd label (False or True)
        :return: ids (int array)       : integer array of ann ids
        """
        img_ids = img_ids if img_ids is not None else []
        cat_ids = cat_ids if cat_ids is not None else []
        area_rng = area_rng if area_rng is not None else []
        img_ids = img_ids if _is_array_like(img_ids) else [img_ids]
        cat_ids = cat_ids if _is_array_like(cat_ids) else [cat_ids]

        if len(img_ids) == len(cat_ids) == len(area_rng) == 0:
            anns = self.dataset["annotations"]
        else:
            if len(img_ids) != 0:
                lists = [self.img_to_anns[imgId] for imgId in img_ids if imgId in self.img_to_anns]
                anns = list(itertools.chain.from_iterable(lists))
            else:
                anns = self.dataset["annotations"]
            anns = (
                anns
                if len(cat_ids) == 0
                else [ann for ann in anns if ann["category_id"] in cat_ids]
            )
            anns = (
                anns
                if len(area_rng) == 0
                else [
                    ann for ann in anns if ann["area"] > area_rng[0] and ann["area"] < area_rng[1]
                ]
            )
        if iscrowd:
            ids = [ann["id"] for ann in anns if bool(ann["iscrowd"]) == bool(iscrowd)]
        else:
            ids = [ann["id"] for ann in anns]
        return ids

    def get_cat_ids(self, cat_nms=None, sup_nms=None, cat_ids=None):
        """
        filtering parameters. default skips that filter.
        :param cat_nms (str array)  : get cats for given cat names
        :param sup_nms (str array)  : get cats for given supercategory names
        :param cat_ids (int array)  : get cats for given cat ids
        :return: ids (int array)   : integer array of cat ids
        """
        cat_nms = cat_nms if cat_nms is not None else []
        sup_nms = sup_nms if sup_nms is not None else []
        cat_ids = cat_ids if cat_ids is not None else []
        cat_nms = cat_nms if _is_array_like(cat_nms) else [cat_nms]
        sup_nms = sup_nms if _is_array_like(sup_nms) else [sup_nms]
        cat_ids = cat_ids if _is_array_like(cat_ids) else [cat_ids]

        if len(cat_nms) == len(sup_nms) == len(cat_ids) == 0:
            cats = self.dataset["categories"]
        else:
            cats = self.dataset["categories"]
            cats = cats if len(cat_nms) == 0 else [cat for cat in cats if cat["name"] in cat_nms]
            cats = (
                cats
                if len(sup_nms) == 0
                else [cat for cat in cats if cat["supercategory"] in sup_nms]
            )
            cats = cats if len(cat_ids) == 0 else [cat for cat in cats if cat["id"] in cat_ids]
        ids = [cat["id"] for cat in cats]
        return ids

    def get_img_ids(self, img_ids=None, cat_ids=None):
        """
        Get img ids that satisfy given filter conditions.
        :param img_ids (int array) : get imgs for given ids
        :param cat_ids (int array) : get imgs with all given cats
        :return: ids (int array)  : integer array of img ids
        """
        img_ids = img_ids if img_ids is not None else []
        cat_ids = cat_ids if cat_ids is not None else []
        img_ids = img_ids if _is_array_like(img_ids) else [img_ids]
        cat_ids = cat_ids if _is_array_like(cat_ids) else [cat_ids]

        if len(img_ids) == len(cat_ids) == 0:
            ids = self.imgs.keys()
        else:
            ids = set(img_ids)
            for i, cat_id in enumerate(cat_ids):
                if i == 0 and len(ids) == 0:
                    ids = set(self.cat_to_imgs[cat_id])
                else:
                    ids &= set(self.cat_to_imgs[cat_id])
        return list(ids)

    def load_anns(self, ids=None):
        """
        Load anns with the specified ids.
        :param ids (int array)       : integer ids specifying anns
        :return: anns (object array) : loaded ann objects
        """
        if _is_array_like(ids):
            return [self.anns[id] for id in ids]
        if isinstance(ids, int):
            return [self.anns[ids]]
        return []

    def load_cats(self, ids=None):
        """
        Load cats with the specified ids.
        :param ids (int array)       : integer ids specifying cats
        :return: cats (object array) : loaded cat objects
        """
        if _is_array_like(ids):
            return [self.cats[id] for id in ids]
        if isinstance(ids, int):
            return [self.cats[ids]]
        return []

    def load_imgs(self, ids=None):
        """
        Load anns with the specified ids.
        :param ids (int array)       : integer ids specifying img
        :return: imgs (object array) : loaded img objects
        """
        if _is_array_like(ids):
            return [self.imgs[id] for id in ids]
        if isinstance(ids, int):
            return [self.imgs[ids]]
        return []

    def get_img_path(self, img_name, pre_path=None):
        """
        Generate a absolute path for single image.
        Default using the validation image path.
        """
        if pre_path is None:
            pre_path = f"{os.environ['ZHOUYI_DATASET_HOME']}/MSCOCO/data/coco2017/val2017"
        return os.path.join(pre_path, img_name)

    def get_label_mapping_dict(self):
        """
        Return a dict that key is index number from model,
        and the value is label in dataset.
        """
        cats = list(self.cats.keys())
        cats.sort()
        model_label_to_coco_label = {}
        for idx, cat in enumerate(cats):
            model_label_to_coco_label[idx] = cat
        return model_label_to_coco_label


class VOC:
    """
    Helper class to get Pascal VOC dataset info
    """

    CLASSES = (
        "aeroplane",
        "bicycle",
        "bird",
        "boat",
        "bottle",
        "bus",
        "car",
        "cat",
        "chair",
        "cow",
        "diningtable",
        "dog",
        "horse",
        "motorbike",
        "person",
        "pottedplant",
        "sheep",
        "sofa",
        "train",
        "tvmonitor",
    )

    def __init__(
        self,
        root_dir=None,
        image_sets=None,
        keep_difficult=False,
    ):
        """
        Constructor of PASCAL VOC helper class for reading and parsing xml file of img.
        :param root_dir (str): Path to folder storing the dataset.
        :param image_sets (str): list of tuples, default [(2007, 'test')].
        :param keep_difficult (bool): keep pictures marked as difficult in the dataset,
        default False.
        :return:
        """
        if root_dir is None:
            root_dir = f"{os.environ['ZHOUYI_DATASET_HOME']}/VOCdevkit/data/"
        assert os.path.isdir(root_dir), f"{root_dir} is not exists."
        if image_sets is None:
            image_sets = [("2007", "test")]
        self._root_dir = root_dir
        self._image_sets = image_sets
        self._keep_difficult = keep_difficult
        self._ids = []
        self._annopath = os.path.join("%s", "Annotations", "%s.xml")
        self._imgpath = os.path.join("%s", "JPEGImages", "%s.jpg")
        self.imgs = dict()
        self.voc_label_to_category_id, self.category_id_to_voc_label = dict(), dict()
        self.img_to_anns, self.cat_to_imgs = defaultdict(list), defaultdict(list)
        print("Start to load voc dataset...")
        tic = time.time()
        self._load_categories()
        self._load_items()
        self._create_index()
        print("Done (t={:0.2f}s)".format(time.time() - tic))

    def _load_categories(self):
        """Load categories"""
        # key: label name, value: int id
        self.voc_label_to_category_id = dict(zip(self.CLASSES, range(len(self.CLASSES))))
        # key: int id, value: label name
        self.category_id_to_voc_label = {v: k for k, v in self.voc_label_to_category_id.items()}

    def _load_items(self):
        """Load individual image indices from image_sets."""
        for (year, name) in self._image_sets:
            rootpath = os.path.join(self._root_dir, "voc" + year)
            lf = os.path.join(rootpath, "ImageSets", "Main", name + ".txt")
            with open(lf, "r") as f:
                self._ids += [(rootpath, line.strip()) for line in f.readlines()]

    def _create_index(self):
        """
        Parsing the xml file of each img, get annotation info.
        """
        print("creating index...")
        imgs = {}
        img_to_anns, cat_to_imgs = defaultdict(list), defaultdict(list)

        for voc_path, img_id in self._ids:
            img_id_int = int(img_id)
            target = ET.parse(self._annopath % (voc_path, img_id)).getroot()
            for obj in target.iter("object"):
                difficult = int(obj.find("difficult").text) == 1
                if not self._keep_difficult and difficult:
                    continue
                name = obj.find("name").text.lower().strip()
                bbox = obj.find("bndbox")
                pts = ["xmin", "ymin", "xmax", "ymax"]
                bndbox = []
                for pt in pts:
                    cur_pt = float(bbox.find(pt).text)
                    bndbox.append(cur_pt)
                label_idx = self.voc_label_to_category_id[name]
                ann = {
                    "bbox": bndbox,  # xmin, ymin, xmax, ymax
                    "category_id": label_idx,  # int id
                }
                img_to_anns[img_id_int].append(ann)
                cat_to_imgs[label_idx].append(img_id_int)

            size = target.find("size")
            width = float(size.find("width").text)
            height = float(size.find("height").text)
            depth = float(size.find("depth").text)
            file_path = self._imgpath % (voc_path, img_id)
            img = {
                "file_path": file_path,
                "height": height,
                "width": width,
                "depth": depth,
            }
            imgs[img_id_int] = img

        print("index created!")
        self.img_to_anns = img_to_anns
        self.cat_to_imgs = cat_to_imgs
        self.imgs = imgs

    def load_anns(self, img_ids=None, cat_ids=None):
        """
        Load anns with the specified ids.
        :param img_ids  (int array)     : get anns for given imgs
               cat_ids  (int array)     : get anns for given cats
        :return: anns (object array) : loaded ann objects
        """
        img_ids = img_ids if img_ids is not None else []
        cat_ids = cat_ids if cat_ids is not None else []
        img_ids = img_ids if _is_array_like(img_ids) else [img_ids]
        cat_ids = cat_ids if _is_array_like(cat_ids) else [cat_ids]

        if len(img_ids) == 0:
            return []

        lists = [self.img_to_anns[imgId] for imgId in img_ids if imgId in self.img_to_anns]
        anns = list(itertools.chain.from_iterable(lists))

        anns = anns if len(cat_ids) == 0 else [ann for ann in anns if ann["category_id"] in cat_ids]

        return anns

    def get_img_ids(self, img_ids=None, cat_ids=None):
        """
        Get img ids that satisfy given filter conditions.
        :param img_ids (int array) : get imgs for given ids
        :param cat_ids (int array) : get imgs with all given cats
        :return: ids (int array)  : integer array of img ids
        """
        img_ids = img_ids if img_ids is not None else []
        cat_ids = cat_ids if cat_ids is not None else []
        img_ids = img_ids if _is_array_like(img_ids) else [img_ids]
        cat_ids = cat_ids if _is_array_like(cat_ids) else [cat_ids]

        if len(img_ids) == len(cat_ids) == 0:
            ids = self.imgs.keys()
        else:
            ids = set(img_ids)
            for i, cat_id in enumerate(cat_ids):
                if i == 0 and len(ids) == 0:
                    ids = set(self.cat_to_imgs[cat_id])
                else:
                    ids &= set(self.cat_to_imgs[cat_id])
        return list(ids)

    def load_cats(self, ids=None):
        """
        Load cats with the specified ids.
        :param ids (int array)       : integer ids specifying cats
        :return: cats (object array) : loaded cat objects
        """
        if _is_array_like(ids):
            return [self.category_id_to_voc_label[id] for id in ids]
        if isinstance(ids, int):
            return [self.category_id_to_voc_label[ids]]
        return []

    def load_imgs(self, ids=None):
        """
        Load imgs with the specified ids.
        :param ids (int array)       : integer ids specifying img
        :return: imgs (object array) : loaded img objects
        """
        if _is_array_like(ids):
            return [self.imgs[id] for id in ids]
        if isinstance(ids, int):
            return [self.imgs[ids]]
        return []


class ClassificationDataset(metaclass=abc.ABCMeta):
    """
    A class to generate the Validation Dataset for Classification Model.

    Methods
    -------
    get_img_path(img_name=""):
        Generate a absolute path for single image.
    get_img_label(img_name=""):
        Return a label id corresponding to the image.
    get_img_list(img_number=100):
        Randomly generate a list of image names according to the specified number.
    """

    @abc.abstractmethod
    def __init__(self):
        """Constructor."""

    @abc.abstractmethod
    def get_img_path(self, img_name):
        """Takes in image name, returns the absolute path of the image"""

    @abc.abstractmethod
    def get_img_label(self, img_name):
        """Takes in image name, returns the coccrect Label ID corresponding to the image."""

    @abc.abstractmethod
    def get_img_list(self, img_number):
        """Takes in image number, returns a list of the corresponding number of image names."""


class ImageNetVal(ClassificationDataset):
    """
    Helper class to get ImageNet Validation dataset info.
    """

    def __init__(self):
        """
        Generate a dictionary of one-to-one correspondence between images and label ids.
        """
        super().__init__()
        val_label_file = f"{os.environ['ZHOUYI_DATASET_HOME']}/ImageNet/data/ILSVRC2012_val.txt"
        self.img_id_mapping_dict = dict()
        if os.path.isfile(val_label_file):
            with open(val_label_file, "r") as f:
                for line in f.readlines():
                    img_name, label_id = line.strip().split()
                    self.img_id_mapping_dict[img_name] = label_id
        else:
            raise FileNotFoundError(f"{val_label_file} is not exists!")

    def get_img_path(self, img_name=""):
        """
        Generate a absolute path for single image.
        """
        if not img_name:
            raise RuntimeError("image name is NOT Valid.")
        img_pre_path = f"{os.environ['ZHOUYI_DATASET_HOME']}/ImageNet/data/ILSVRC2012_img_val"
        return os.path.join(img_pre_path, img_name)

    def get_img_label(self, img_name=""):
        """
        Return a label id corresponding to the image.
        """
        # The label index in synset is start from 1,
        # So here need to add 1 to the index value,
        # because the index of img's label starts from 0
        if not img_name:
            raise RuntimeError("image name is NOT Valid.")
        return int(self.img_id_mapping_dict[img_name]) + 1

    def get_img_list(self, img_number=100):
        """
        Randomly generate a list of image names according to the specified number.
        """
        if img_number < 1 or img_number > 50000:
            raise RuntimeError("The number of image must be between 1 and 50000.")
        random.seed(1)  # set seed to 1 for keeping random image datasets consistent
        return random.sample(list(self.img_id_mapping_dict), img_number)
