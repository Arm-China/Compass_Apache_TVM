# This file is CONFIDENTIAL and created by Arm Technology (China) Co., Ltd.
# See the copyright file distributed with this work for additional information
# regarding copyright ownership.
# pylint: disable=invalid-name
"""Module for processing the data related to models."""
import math
import numpy as np
from scipy import special


def draw_box(draw, box, thickness=2, color="red", label=""):
    """Draw rectangles on pictures and add labels."""
    top = int(box[0])
    left = int(box[1])
    bottom = int(box[2])
    right = int(box[3])

    draw.rectangle(
        [left + thickness, top + thickness, right - thickness, bottom - thickness], outline=color
    )

    label_size = draw.textsize(label)
    if top - label_size[1] >= 0:
        text_origin = tuple(np.array([left, top - label_size[1]]))
    else:
        text_origin = tuple(np.array([left, top + 1]))
    draw.text(text_origin, label, fill=color)


def _bboxes_sort(classes, scores, bboxes, top_k=400):
    index = np.argsort(-scores)
    classes = classes[index][:top_k]
    scores = scores[index][:top_k]
    bboxes = bboxes[index][:top_k]
    return classes, scores, bboxes


def _bboxes_iou(bboxes1, bboxes2):
    bboxes1 = np.transpose(bboxes1)
    bboxes2 = np.transpose(bboxes2)

    int_ymin = np.maximum(bboxes1[0], bboxes2[0])
    int_xmin = np.maximum(bboxes1[1], bboxes2[1])
    int_ymax = np.minimum(bboxes1[2], bboxes2[2])
    int_xmax = np.minimum(bboxes1[3], bboxes2[3])

    int_h = np.maximum(int_ymax - int_ymin, 0.0)
    int_w = np.maximum(int_xmax - int_xmin, 0.0)

    int_vol = int_h * int_w
    vol1 = (bboxes1[2] - bboxes1[0]) * (bboxes1[3] - bboxes1[1])
    vol2 = (bboxes2[2] - bboxes2[0]) * (bboxes2[3] - bboxes2[1])
    IOU = int_vol / (vol1 + vol2 - int_vol)
    return IOU


def NMS(classes, scores, bboxes, threshold=0.5):
    """Process NMS for boxes."""
    # Sort and clip boxes(default 400)
    classes, scores, bboxes = _bboxes_sort(classes, scores, bboxes)

    # NMS
    keep_bboxes = np.ones(scores.shape, dtype=np.bool_)
    for i in range(scores.size - 1):
        if keep_bboxes[i]:
            overlap = _bboxes_iou(bboxes[i], bboxes[(i + 1) :])
            keep_overlap = np.logical_or(overlap < threshold, classes[(i + 1) :] != classes[i])
            keep_bboxes[(i + 1) :] = np.logical_and(keep_bboxes[(i + 1) :], keep_overlap)

    idxes = np.where(keep_bboxes)
    return classes[idxes], scores[idxes], bboxes[idxes]


class PostprocessYOLOv2(object):
    """Class for post-processing the outputs from YOLOv2-416."""

    def __init__(
        self,
        anchors,
        image_shape,
        num_categories,
        obj_threshold=0.5,
        nms_threshold=0.5,
    ):
        self.anchors = anchors
        self.image_shape = image_shape
        self.num_categories = num_categories
        self.obj_threshold = obj_threshold
        self.nms_threshold = nms_threshold

    def process(self, model_output):
        """Take the YOLOv2 outputs, post-process them
        and return a list of bounding boxes for detected object
        together with their category
        and their confidences in separate lists.
        """
        # Convert output to numpy format
        model_output = model_output[0].numpy()
        # Decode boxes, scores and classes from outputs
        bboxes, obj_probs, class_probs = self._decode(
            model_output, self.num_categories, self.anchors
        )
        # Filter the decoded regression bounding box
        bboxes, scores, classes = self._postprocess(
            bboxes, obj_probs, class_probs, image_shape=self.image_shape
        )
        # Adjust the element position of the boxes
        for box in bboxes:
            box[0], box[1], box[2], box[3] = box[1], box[0], box[3], box[2]
        # Convert class number to int type
        classes_int = [int(clas) for clas in classes]
        return bboxes, scores, classes_int

    def _decode(self, model_output, num_class=80, anchors=None):
        # Reshape model_output from nchw to nhwc format.
        if len(model_output.shape) == 4:
            model_output = np.transpose(model_output, [0, 2, 3, 1])

        H, W = model_output.shape[1:3]
        num_anchors = len(anchors)
        anchors = np.array(anchors, dtype=np.float32)

        # Reshape output to 13*13*num_anchors*(num_class+5), the first dimension adaptive batchsize
        detection_result = np.reshape(model_output, [-1, H * W, num_anchors, num_class + 5])

        def sigmoid(value):
            # Return the sigmoid of the input.
            return 1.0 / (1.0 + math.exp(-value))

        def exponential(value):
            # Return the exponential of the input.
            return math.exp(value)

        sigmoid_v = np.vectorize(sigmoid)
        exponential_v = np.vectorize(exponential)

        # Output transformation - offset, confidence, class probability
        xy_offset = sigmoid_v(detection_result[:, :, :, 0:2])
        wh_offset = exponential_v(detection_result[:, :, :, 2:4])
        obj_probs = sigmoid_v(detection_result[:, :, :, 4])
        class_probs = sigmoid_v(detection_result[:, :, :, 5:])

        # Build the xy coordinates of the upper left corner of each cell in the feature map
        height_index = np.arange(H, dtype=np.float32)
        width_index = np.arange(W, dtype=np.float32)
        x_cell, y_cell = np.meshgrid(height_index, width_index)
        x_cell = np.reshape(x_cell, [1, -1, 1])
        y_cell = np.reshape(y_cell, [1, -1, 1])

        # Decode
        bbox_x = (x_cell + xy_offset[:, :, :, 0]) / W
        bbox_y = (y_cell + xy_offset[:, :, :, 1]) / H
        bbox_w = (anchors[:, 0] * wh_offset[:, :, :, 0]) / W
        bbox_h = (anchors[:, 1] * wh_offset[:, :, :, 1]) / H

        # Center coordinates + width and height box(x,y,w,h)
        # -> xmin=x-w/2
        # -> upper left + lower right box(xmin,ymin,xmax,ymax)
        bboxes = np.stack(
            [bbox_x - bbox_w / 2, bbox_y - bbox_h / 2, bbox_x + bbox_w / 2, bbox_y + bbox_h / 2],
            axis=3,
        )

        return bboxes, obj_probs, class_probs

    def _postprocess(self, bboxes, obj_probs, class_probs, image_shape=(416, 416)):
        bboxes = np.reshape(bboxes, [-1, 4])  # xmin,ymin,xmax,ymax
        # Scale boxes back to original image shape
        bboxes[:, 0:1] *= float(image_shape[1])  # xmin*width
        bboxes[:, 1:2] *= float(image_shape[0])  # ymin*height
        bboxes[:, 2:3] *= float(image_shape[1])  # xmax*width
        bboxes[:, 3:4] *= float(image_shape[0])  # ymax*height
        bboxes = bboxes.astype(np.int32)

        # (1) cut the box that are out of range
        bbox_min_max = [0, 0, image_shape[1] - 1, image_shape[0] - 1]
        bboxes = self._bboxes_cut(bbox_min_max, bboxes)

        # Confidence*max class probability = class confidence scores
        obj_probs = np.reshape(obj_probs, [-1])
        class_probs = np.reshape(class_probs, [len(obj_probs), -1])
        class_max_index = np.argmax(class_probs, axis=1)
        class_probs = class_probs[np.arange(len(obj_probs)), class_max_index]
        scores = obj_probs * class_probs

        # Discard some boxes with low scores
        keep_index = scores > self.obj_threshold
        class_max_index = class_max_index[keep_index]
        scores = scores[keep_index]
        bboxes = bboxes[keep_index]

        # NMS
        class_max_index, scores, bboxes = NMS(
            class_max_index, scores, bboxes, threshold=self.nms_threshold
        )

        return bboxes, scores, class_max_index

    def _bboxes_cut(self, bbox_min_max, bboxes):
        bboxes = np.copy(bboxes)
        bboxes = np.transpose(bboxes)
        bbox_min_max = np.transpose(bbox_min_max)
        bboxes[0] = np.maximum(bboxes[0], bbox_min_max[0])  # xmin
        bboxes[1] = np.maximum(bboxes[1], bbox_min_max[1])  # ymin
        bboxes[2] = np.minimum(bboxes[2], bbox_min_max[2])  # xmax
        bboxes[3] = np.minimum(bboxes[3], bbox_min_max[3])  # ymax
        bboxes = np.transpose(bboxes)
        return bboxes


class PostprocessYOLOv3(object):
    """Class for post-processing the outputs from YOLOv3-416."""

    def __init__(self, image_shape):
        self.image_shape = image_shape

    def process(self, model_output):
        """Take the YOLOv3 outputs, post-process them
        and return a list of bounding boxes for detected object
        together with their category
        and their confidences in separate lists.
        """
        # Convert output to numpy format
        boxes, scores, indices = (x.numpy() for x in model_output)

        out_boxes, out_scores, out_classes = [], [], []
        for idx_ in indices:
            out_classes.append(idx_[1])
            out_scores.append(scores[tuple(idx_)])
            idx_1 = (idx_[0], idx_[2])
            out_boxes.append(boxes[idx_1])

        origin_h, origin_w = self.image_shape
        out_boxes = [self._map_resized_box(origin_h, origin_w, 416, 416, box) for box in out_boxes]

        return out_boxes, out_scores, out_classes

    def _map_resized_box(self, origin_h, origin_w, resized_h, resized_w, box):
        iw, ih = origin_w, origin_h
        w, h = resized_w, resized_h
        scale = min(w / iw, h / ih)
        nw = int(iw * scale)
        nh = int(ih * scale)
        pad_h = resized_h - nh
        pad_w = resized_w - nw
        border_top = pad_h // 2
        border_left = pad_w // 2

        top = (box[0] - border_top) / scale
        left = (box[1] - border_left) / scale
        bottom = (box[2] - border_top) / scale
        right = (box[3] - border_left) / scale
        return [top, left, bottom, right]


class PostprocessYOLOv3_608(object):
    """Class for post-processing the three outputs from YOLOv3-608."""

    def __init__(self, yolo_masks, yolo_anchors, obj_threshold, nms_threshold, model_input_hw):

        self.masks = yolo_masks
        self.anchors = yolo_anchors
        self.object_threshold = obj_threshold
        self.nms_threshold = nms_threshold
        self.model_input_hw = model_input_hw

    def process(self, outputs, image_raw_size, classes=80):
        """Take three outputs:(1, 255, 19, 19), (1, 255, 38, 38), (1, 255, 76, 76)
        from yolov3(608), return boxes, categories, confidences for detected object.
        """
        outputs = [x.numpy() for x in outputs]
        outputs_reshaped = []
        for output in outputs:
            outputs_reshaped.append(self._reshape_output(output, classes))

        boxes, categories, confidences = self._process_yolo_output(outputs_reshaped, image_raw_size)

        for bbox in boxes:
            bbox[0], bbox[1], bbox[2], bbox[3] = (
                bbox[1],
                bbox[0],
                bbox[1] + bbox[3],
                bbox[0] + bbox[2],
            )

        return boxes, categories, confidences

    def _reshape_output(self, output, classes):
        """(1, 255, x, x) -> (x, x, 3, 85)"""
        output = np.transpose(output, [0, 2, 3, 1])
        _, height, width, _ = output.shape
        dim1, dim2 = height, width
        dim3 = 3
        dim4 = 4 + 1 + classes
        return np.reshape(output, (dim1, dim2, dim3, dim4))

    def _process_yolo_output(self, outputs_reshaped, image_raw_size):
        boxes, categories, confidences = [], [], []
        for output, mask in zip(outputs_reshaped, self.masks):
            box, category, confidence = self._process_feats(output, mask)
            box, category, confidence = self._filter_boxes(box, category, confidence)
            boxes.append(box)
            categories.append(category)
            confidences.append(confidence)

        boxes = np.concatenate(boxes)
        categories = np.concatenate(categories)
        confidences = np.concatenate(confidences)

        # Scale boxes back to original image shape:
        width, height = image_raw_size
        image_dims = [width, height, width, height]
        boxes = boxes * image_dims

        # Using the candidates from the previous (loop) step, we apply the non-max suppression
        # algorithm that clusters adjacent bounding boxes to a single bounding box:
        nms_boxes, nms_categories, nscores = [], [], []
        for category in set(categories):
            idxs = np.where(categories == category)
            box = boxes[idxs]
            category = categories[idxs]
            confidence = confidences[idxs]

            keep = self._nms_boxes(box, confidence)

            nms_boxes.append(box[keep])
            nms_categories.append(category[keep])
            nscores.append(confidence[keep])

        if not nms_categories and not nscores:
            return [], [], []

        boxes = np.concatenate(nms_boxes)
        categories = np.concatenate(nms_categories)
        confidences = np.concatenate(nscores)

        return boxes, categories, confidences

    def _process_feats(self, output_reshaped, mask):
        # Two in-line functions required for calculating the bounding box
        # descriptors:
        def sigmoid(value):
            """Return the sigmoid of the input."""
            return 1.0 / (1.0 + math.exp(-value))

        def exponential(value):
            """Return the exponential of the input."""
            return math.exp(value)

        # Vectorized calculation of above two functions:
        sigmoid_v = np.vectorize(sigmoid)
        exponential_v = np.vectorize(exponential)

        grid_h, grid_w, _, _ = output_reshaped.shape

        anchors = [self.anchors[i] for i in mask]

        # Reshape to N, height, width, num_anchors, box_params:
        anchors_tensor = np.reshape(anchors, [1, 1, len(anchors), 2])
        box_xy = sigmoid_v(output_reshaped[..., :2])
        box_wh = exponential_v(output_reshaped[..., 2:4]) * anchors_tensor
        box_confidence = sigmoid_v(output_reshaped[..., 4])

        box_confidence = np.expand_dims(box_confidence, axis=-1)
        box_class_probs = sigmoid_v(output_reshaped[..., 5:])

        col = np.tile(np.arange(0, grid_w), grid_w).reshape(-1, grid_w)
        row = np.tile(np.arange(0, grid_h).reshape(-1, 1), grid_h)

        col = col.reshape(grid_h, grid_w, 1, 1).repeat(3, axis=-2)
        row = row.reshape(grid_h, grid_w, 1, 1).repeat(3, axis=-2)
        grid = np.concatenate((col, row), axis=-1)

        box_xy += grid
        box_xy /= (grid_w, grid_h)
        box_wh /= self.model_input_hw
        box_xy -= box_wh / 2.0
        boxes = np.concatenate((box_xy, box_wh), axis=-1)

        return boxes, box_confidence, box_class_probs

    def _filter_boxes(self, boxes, box_confidences, box_class_probs):
        box_scores = box_confidences * box_class_probs
        box_classes = np.argmax(box_scores, axis=-1)
        box_class_scores = np.max(box_scores, axis=-1)
        pos = np.where(box_class_scores >= self.object_threshold)

        boxes = boxes[pos]
        classes = box_classes[pos]
        scores = box_class_scores[pos]

        return boxes, classes, scores

    def _nms_boxes(self, boxes, box_confidences):
        x_coord = boxes[:, 0]
        y_coord = boxes[:, 1]
        width = boxes[:, 2]
        height = boxes[:, 3]

        areas = width * height
        ordered = box_confidences.argsort()[::-1]

        keep = []
        while ordered.size > 0:
            # Index of the current element:
            i = ordered[0]
            keep.append(i)
            xx1 = np.maximum(x_coord[i], x_coord[ordered[1:]])
            yy1 = np.maximum(y_coord[i], y_coord[ordered[1:]])
            xx2 = np.minimum(x_coord[i] + width[i], x_coord[ordered[1:]] + width[ordered[1:]])
            yy2 = np.minimum(y_coord[i] + height[i], y_coord[ordered[1:]] + height[ordered[1:]])

            width1 = np.maximum(0.0, xx2 - xx1 + 1)
            height1 = np.maximum(0.0, yy2 - yy1 + 1)
            intersection = width1 * height1
            union = areas[i] + areas[ordered[1:]] - intersection

            iou = intersection / union
            indexes = np.where(iou <= self.nms_threshold)[0]
            ordered = ordered[indexes + 1]

        keep = np.array(keep)
        return keep


class PostprocessYOLOv4(object):
    """Class for post-processing the outputs from YOLOv4-416."""

    def __init__(
        self,
        anchors,
        image_shape,
        num_categories,
        obj_threshold=0.25,
        nms_threshold=0.213,
    ):
        self.anchors = anchors
        self.image_shape = image_shape
        self.num_categories = num_categories
        self.obj_threshold = obj_threshold
        self.nms_threshold = nms_threshold

    def process(self, model_output):
        """Take the YOLOv4 outputs, post-process them
        and return a list of bounding boxes for detected object
        together with their category
        and their confidences in in separate lists.
        """
        # Convert output to numpy format
        detections = [x.numpy() for x in model_output]
        ANCHORS = np.array(self.anchors)
        STRIDES = np.array([8, 16, 32])
        XYSCALE = [1.2, 1.1, 1.05]
        # Decode boxes, scores and classes from outputs
        pred_bbox = self._postprocess_bbbox(detections, ANCHORS, STRIDES, XYSCALE)
        # Filter the decoded regression bounding box
        bboxes = self._postprocess_boxes(pred_bbox, self.image_shape, 416, self.obj_threshold)
        bboxes = self._nms(bboxes, self.nms_threshold, method="nms")
        out_boxes, out_scores, out_classes = [], [], []
        # bboxes: [x_min, y_min, x_max, y_max, probability, cls_id] format coordinates.
        for bbox in bboxes:
            # Adjust the element position of the boxes
            box = bbox[:4]
            box[0], box[1], box[2], box[3] = box[1], box[0], box[3], box[2]
            out_boxes.append(box)
            out_scores.append(bbox[4])
            out_classes.append(int(bbox[5]))

        return out_boxes, out_scores, out_classes

    def _postprocess_bbbox(self, pred_bbox, ANCHORS, STRIDES, XYSCALE):
        """define anchor boxes"""
        for i, pred in enumerate(pred_bbox):
            conv_shape = pred.shape
            output_size = conv_shape[1]
            conv_raw_dxdy = pred[:, :, :, :, 0:2]
            conv_raw_dwdh = pred[:, :, :, :, 2:4]
            xy_grid = np.meshgrid(np.arange(output_size), np.arange(output_size))
            xy_grid = np.expand_dims(np.stack(xy_grid, axis=-1), axis=2)

            xy_grid = np.tile(np.expand_dims(xy_grid, axis=0), [1, 1, 1, 3, 1])
            xy_grid = xy_grid.astype(np.float)

            pred_xy = (
                (special.expit(conv_raw_dxdy) * XYSCALE[i]) - 0.5 * (XYSCALE[i] - 1) + xy_grid
            ) * STRIDES[i]
            pred_wh = np.exp(conv_raw_dwdh) * ANCHORS[i]
            pred[:, :, :, :, 0:4] = np.concatenate([pred_xy, pred_wh], axis=-1)

        pred_bbox = [np.reshape(x, (-1, np.shape(x)[-1])) for x in pred_bbox]
        pred_bbox = np.concatenate(pred_bbox, axis=0)
        return pred_bbox

    def _postprocess_boxes(self, pred_bbox, org_img_shape, input_size, score_threshold):
        """remove boundary boxs with a low detection probability"""
        valid_scale = [0, np.inf]
        pred_bbox = np.array(pred_bbox)

        pred_xywh = pred_bbox[:, 0:4]
        pred_conf = pred_bbox[:, 4]
        pred_prob = pred_bbox[:, 5:]

        # # (1) (x, y, w, h) --> (xmin, ymin, xmax, ymax)
        pred_coor = np.concatenate(
            [pred_xywh[:, :2] - pred_xywh[:, 2:] * 0.5, pred_xywh[:, :2] + pred_xywh[:, 2:] * 0.5],
            axis=-1,
        )
        # # (2) (xmin, ymin, xmax, ymax) -> (xmin_org, ymin_org, xmax_org, ymax_org)
        org_h, org_w = org_img_shape
        resize_ratio = min(input_size / org_w, input_size / org_h)

        dw = (input_size - resize_ratio * org_w) / 2
        dh = (input_size - resize_ratio * org_h) / 2

        pred_coor[:, 0::2] = 1.0 * (pred_coor[:, 0::2] - dw) / resize_ratio
        pred_coor[:, 1::2] = 1.0 * (pred_coor[:, 1::2] - dh) / resize_ratio

        # # (3) clip some boxes that are out of range
        pred_coor = np.concatenate(
            [
                np.maximum(pred_coor[:, :2], [0, 0]),
                np.minimum(pred_coor[:, 2:], [org_w - 1, org_h - 1]),
            ],
            axis=-1,
        )
        invalid_mask = np.logical_or(
            (pred_coor[:, 0] > pred_coor[:, 2]), (pred_coor[:, 1] > pred_coor[:, 3])
        )
        pred_coor[invalid_mask] = 0

        # # (4) discard some invalid boxes
        bboxes_scale = np.sqrt(np.multiply.reduce(pred_coor[:, 2:4] - pred_coor[:, 0:2], axis=-1))
        scale_mask = np.logical_and(
            (valid_scale[0] < bboxes_scale), (bboxes_scale < valid_scale[1])
        )

        # # (5) discard some boxes with low scores
        classes = np.argmax(pred_prob, axis=-1)
        scores = pred_conf * pred_prob[np.arange(len(pred_coor)), classes]
        score_mask = scores > score_threshold
        mask = np.logical_and(scale_mask, score_mask)
        coors, scores, classes = pred_coor[mask], scores[mask], classes[mask]

        return np.concatenate([coors, scores[:, np.newaxis], classes[:, np.newaxis]], axis=-1)

    def _bboxes_iou(self, boxes1, boxes2):
        """calculate the Intersection Over Union value"""
        boxes1 = np.array(boxes1)
        boxes2 = np.array(boxes2)

        boxes1_area = (boxes1[..., 2] - boxes1[..., 0]) * (boxes1[..., 3] - boxes1[..., 1])
        boxes2_area = (boxes2[..., 2] - boxes2[..., 0]) * (boxes2[..., 3] - boxes2[..., 1])

        left_up = np.maximum(boxes1[..., :2], boxes2[..., :2])
        right_down = np.minimum(boxes1[..., 2:], boxes2[..., 2:])

        inter_section = np.maximum(right_down - left_up, 0.0)
        inter_area = inter_section[..., 0] * inter_section[..., 1]
        union_area = boxes1_area + boxes2_area - inter_area
        ious = np.maximum(1.0 * inter_area / union_area, np.finfo(np.float32).eps)

        return ious

    def _nms(self, bboxes, iou_threshold, sigma=0.3, method="nms"):
        """
        :param bboxes: (xmin, ymin, xmax, ymax, score, class)

        Note: soft-nms, https://arxiv.org/pdf/1704.04503.pdf
            https://github.com/bharatsingh430/soft-nms
        """
        classes_in_img = list(set(bboxes[:, 5]))
        best_bboxes = []

        for cls in classes_in_img:
            cls_mask = bboxes[:, 5] == cls
            cls_bboxes = bboxes[cls_mask]

            while len(cls_bboxes) > 0:
                max_ind = np.argmax(cls_bboxes[:, 4])
                best_bbox = cls_bboxes[max_ind]
                best_bboxes.append(best_bbox)
                cls_bboxes = np.concatenate([cls_bboxes[:max_ind], cls_bboxes[max_ind + 1 :]])
                iou = self._bboxes_iou(best_bbox[np.newaxis, :4], cls_bboxes[:, :4])
                weight = np.ones((len(iou),), dtype=np.float32)

                assert method in ["nms", "soft-nms"]

                if method == "nms":
                    iou_mask = iou > iou_threshold
                    weight[iou_mask] = 0.0

                if method == "soft-nms":
                    weight = np.exp(-(1.0 * iou**2 / sigma))

                cls_bboxes[:, 4] = cls_bboxes[:, 4] * weight
                score_mask = cls_bboxes[:, 4] > 0.0
                cls_bboxes = cls_bboxes[score_mask]

        return best_bboxes


class PostprocessSSDMobilenet(object):
    """Class for post-processing the outputs from ssd_mobilenet_v2/v1."""

    def __init__(
        self,
        anchors,
        image_shape,
        num_categories,
        score_threshold=0.5,
        nms_threshold=0.5,
        variance=(10.0, 10.0, 5.0, 5.0),
    ):
        self.anchors = anchors
        self.image_shape = image_shape
        self.num_categories = num_categories
        self.score_threshold = score_threshold
        self.nms_threshold = nms_threshold
        self.variance = variance

    def process(self, model_output):
        """Post-process the ssd_mobilenet_v2 outputs, and
        return bounding boxes, scores, classes.
        """
        # Convert output to numpy format
        box_encode, cls_score = (x.numpy().squeeze() for x in model_output)
        # Decode boxes
        box_decode = self._decode(box_encode, self.anchors)
        # Filter the decoded regression bounding box
        bboxes, scores, classes = self._postprocess(
            box_decode, cls_score, image_shape=self.image_shape
        )

        return bboxes, scores, classes + 1

    def _tile_anchor(
        self,
        grid_height,
        grid_width,
        scales_grid,
        aspect_ratios_grid,
        anchor_stride_,
        anchor_offset_,
        base_anchor_size,
    ):
        """
        get the fixed anchor with the given anchor_scale and anchor_aspect_ratio.
        """

        scales_grid = np.reshape(scales_grid, [-1])
        aspect_ratios_grid = np.reshape(aspect_ratios_grid, [-1])
        ratio_sqrts = np.sqrt(aspect_ratios_grid)

        heights = scales_grid / ratio_sqrts * base_anchor_size[0]
        widths = scales_grid * ratio_sqrts * base_anchor_size[1]

        y_centers = np.array(range(grid_height), dtype=np.float)
        y_centers = y_centers * anchor_stride_[0] + anchor_offset_[0]
        x_centers = np.array(range(grid_width), dtype=np.float)
        x_centers = x_centers * anchor_stride_[1] + anchor_offset_[1]
        x_centers, y_centers = np.meshgrid(x_centers, y_centers)

        widths_grid, x_centers_grid = np.meshgrid(widths, x_centers)
        heights_grid, y_centers_grid = np.meshgrid(heights, y_centers)

        bbox_centers = np.stack(
            [y_centers_grid[:, :, np.newaxis], x_centers_grid[:, :, np.newaxis]], axis=3
        )
        bbox_sizes = np.stack(
            [heights_grid[:, :, np.newaxis], widths_grid[:, :, np.newaxis]], axis=3
        )
        bbox_centers = np.reshape(bbox_centers, [-1, 2])
        bbox_sizes = np.reshape(bbox_sizes, [-1, 2])

        bbox_corners = np.concatenate(
            [bbox_centers - 0.5 * bbox_sizes, bbox_centers + 0.5 * bbox_sizes], 1
        )

        y_min, x_min, y_max, x_max = np.split(bbox_corners, 4, axis=1)

        win_y_min = 0
        win_x_min = 0
        win_y_max = 1.0
        win_x_max = 1.0
        y_min = np.clip(y_min, win_y_min, win_y_max)
        y_max = np.clip(y_max, win_y_min, win_y_max)
        x_min = np.clip(x_min, win_x_min, win_x_max)
        x_max = np.clip(x_max, win_x_min, win_x_max)

        bboxes = np.concatenate([y_min, x_min, y_max, x_max], 1)
        areas = np.squeeze((y_max - y_min) * (x_max - x_min))
        bboxes = bboxes[areas > 0]

        ycenter_a = (bboxes[:, 0] + bboxes[:, 2]) / 2
        xcenter_a = (bboxes[:, 1] + bboxes[:, 3]) / 2
        ha = bboxes[:, 2] - bboxes[:, 0]
        wa = bboxes[:, 3] - bboxes[:, 1]

        return ycenter_a, xcenter_a, ha, wa

    def _gen_anchor_box(self, feature_map_):
        min_scale = 0.2
        max_scale = 0.950000
        num_layers = 6
        anchor_scales_ = np.linspace(min_scale, max_scale, num_layers).tolist()

        append_anchor_scales = anchor_scales_ + [1.0]
        inter_scales = [
            np.sqrt(i * j) for i, j in zip(append_anchor_scales[:-1], append_anchor_scales[1:])
        ]
        firstbox_scale = [10.0, 5.0, 5.0]
        anchor_aspect_ratios_ = [1.0, 2.0, 0.5, 3.0, 0.333333]
        idx = 0
        total_ycenter_a = None
        total_xcenter_a = None
        total_ha = None
        total_wa = None
        for scale, feat_map in zip(anchor_scales_, feature_map_):
            aspect_ratios = []
            scales = []

            if idx == 0:
                scales = 1.0 / np.reshape(firstbox_scale, -1)
                aspect_ratios = [1.0, 2.0, 0.5]
            else:
                for aspect_ratio in anchor_aspect_ratios_:
                    scales.append(scale)
                    aspect_ratios.append(aspect_ratio)
                scales.append(inter_scales[idx])
                aspect_ratios.append(1.0)
            idx += 1

            feat_h, feat_w = feat_map[0], feat_map[1]
            anchor_stride_ = [1.0 / feat_h, 1.0 / feat_w]
            anchor_offset_ = [0.5 * a_s for a_s in anchor_stride_]
            base_anchor_size = [1.0, 1.0]

            ycenter_a, xcenter_a, ha, wa = self._tile_anchor(
                feat_h,
                feat_w,
                scales,
                aspect_ratios,
                anchor_stride_,
                anchor_offset_,
                base_anchor_size,
            )

            if total_ycenter_a is None:
                total_ycenter_a = ycenter_a
                total_xcenter_a = xcenter_a
                total_ha = ha
                total_wa = wa
            else:
                total_ycenter_a = np.concatenate([total_ycenter_a, ycenter_a])
                total_xcenter_a = np.concatenate([total_xcenter_a, xcenter_a])
                total_ha = np.concatenate([total_ha, ha])
                total_wa = np.concatenate([total_wa, wa])
        anchor_box = np.stack(
            [total_ycenter_a, total_xcenter_a, total_ha, total_wa], axis=1
        ).astype(np.float32)
        return anchor_box

    def _decode(self, bbox, anchors):
        anchor_box = self._gen_anchor_box(anchors)
        ty, tx, th, tw = bbox[:, 0], bbox[:, 1], bbox[:, 2], bbox[:, 3]
        variance = self.variance
        ya, xa, ha, wa = anchor_box[:, 0], anchor_box[:, 1], anchor_box[:, 2], anchor_box[:, 3]
        ty = ty / variance[0]
        tx = tx / variance[1]
        th = th / variance[2]
        tw = tw / variance[3]

        h = np.exp(th) * ha
        w = np.exp(tw) * wa
        cy = ty * ha + ya
        cx = tx * wa + xa

        ymin = cy - h / 2.0
        xmin = cx - w / 2.0
        ymax = cy + h / 2.0
        xmax = cx + w / 2.0

        ymin = ymin.clip(0.0, 1.0)
        xmin = xmin.clip(0.0, 1.0)
        ymax = ymax.clip(0.0, 1.0)
        xmax = xmax.clip(0.0, 1.0)

        boxes = np.stack([ymin, xmin, ymax, xmax], axis=-1)
        return boxes

    def _postprocess(self, bboxes, cls_score, image_shape):
        # Scale boxes back to original image shape
        bboxes[:, 0:1] *= float(image_shape[0])  # ymin*height
        bboxes[:, 1:2] *= float(image_shape[1])  # xmin*width
        bboxes[:, 2:3] *= float(image_shape[0])  # ymax*height
        bboxes[:, 3:4] *= float(image_shape[1])  # xmax*width
        bboxes = bboxes.astype(np.int32)

        # exclude the backgroud if have
        bg = cls_score.shape[1] - self.num_categories
        cls_score = cls_score[:, bg:]
        prop_index = np.where(cls_score > self.score_threshold)
        box_ids, class_ids = prop_index[0], prop_index[1]
        bboxes = bboxes[box_ids]
        scores = cls_score[prop_index]

        # NMS
        class_ids, scores, bboxes = NMS(class_ids, scores, bboxes, self.nms_threshold)

        return bboxes, scores, class_ids


class PostprocessSSDResnet(object):
    """Class for post-processing the outputs from ssd_resnet_34."""

    def __init__(
        self,
        image_shape,
        num_categories,
        score_threshold=0.5,
        nms_threshold=0.5,
    ):
        self.image_shape = image_shape
        self.num_categories = num_categories
        self.score_threshold = score_threshold
        self.nms_threshold = nms_threshold

    def process(self, model_output):
        """Post-process the ssd_resnet_34 outputs, and
        return bounding boxes, scores, classes.
        """
        # Convert output to numpy format
        cls_score, box_decode = (x.numpy().squeeze() for x in model_output)
        # Filter the decoded regression bounding box
        bboxes, scores, classes = self._postprocess(
            box_decode, cls_score, image_shape=self.image_shape
        )

        return bboxes, scores, classes

    def _postprocess(self, bboxes, cls_score, image_shape):
        # Scale boxes back to original image shape
        bboxes[:, 0:1] *= float(image_shape[0])  # ymin*height
        bboxes[:, 1:2] *= float(image_shape[1])  # xmin*width
        bboxes[:, 2:3] *= float(image_shape[0])  # ymax*height
        bboxes[:, 3:4] *= float(image_shape[1])  # xmax*width
        bboxes = bboxes.astype(np.int32)

        # exclude the backgroud if have
        bg = cls_score.shape[1] - self.num_categories
        cls_score = cls_score[:, bg:]
        prop_index = np.where(cls_score > self.score_threshold)
        box_ids, class_ids = prop_index[0], prop_index[1]
        bboxes = bboxes[box_ids]
        scores = cls_score[prop_index]

        # NMS
        class_ids, scores, bboxes = NMS(class_ids, scores, bboxes, self.nms_threshold)

        return bboxes, scores, class_ids
