import numpy as np
import tensorflow as tf
import pkg_resources

import cv2


class YOLOV3:
    SCORE_THRESHOLD = 0.3
    IOU_THRESHOLD = 0.45

    def __init__(self):
        label_file = pkg_resources.resource_filename(__name__, 'coco.names')
        self.labels = np.array(open(label_file).read().strip().split("\n"))

        return_elements = ["input/input_data:0",
                           "pred_sbbox/concat_2:0",
                           "pred_mbbox/concat_2:0",
                           "pred_lbbox/concat_2:0"]

        pb_file = pkg_resources.resource_filename(__name__, 'yolov3_coco.pb')
        self.num_classes = 80
        self.input_size = 416
        self.graph = tf.Graph()
        with tf.gfile.FastGFile(pb_file, 'rb') as f:
            frozen_graph_def = tf.GraphDef()
            frozen_graph_def.ParseFromString(f.read())

        with self.graph.as_default():
            self.return_tensors = tf.import_graph_def(frozen_graph_def,
                                                      return_elements=return_elements)
        self.session = tf.Session(graph=self.graph)

        dog_file = pkg_resources.resource_filename(__name__, 'dog.jpg')
        img = cv2.imread(dog_file)
        self.process(img)

    def process(self, img):
        h, w = img.shape[:2]
        image_data = self._preporcess(img)
        pred_sbbox, pred_mbbox, pred_lbbox = self.session.run(
            [self.return_tensors[1], self.return_tensors[2], self.return_tensors[3]],
            feed_dict={self.return_tensors[0]: image_data}
        )

        pred_bbox = np.concatenate([np.reshape(pred_sbbox, (-1, 5 + self.num_classes)),
                                    np.reshape(pred_mbbox, (-1, 5 + self.num_classes)),
                                    np.reshape(pred_lbbox, (-1, 5 + self.num_classes))],
                                   axis=0)

        pred_bboxes = self._postprocess(pred_bbox, h, w)
        pred_bboxes = self._nms(pred_bboxes)
        classes = self.labels[pred_bboxes[:, 5].astype(np.int8)]
        return pred_bboxes, classes

    def _postprocess(self, pred_bbox, org_h, org_w):
        rects = pred_bbox[:, 0:4]  # boxes: centerX, centerY, w, h
        confs = pred_bbox[:, 4]  # confidences
        class_probs = pred_bbox[:, 5:]  # probs for all 80 classes

        # (centerX, centerY, w, h) --> (topLeftX, topLeftY, bottRightX, bottRightY)
        pred_coor = np.c_[rects[:, :2]-rects[:, 2:]*0.5, rects[:, :2]+rects[:, 2:]*0.5]

        # (topLeftX, topLeftY, bottRightX, bottRightY)
        # -> original (topLeftX, topLeftY, bottRightX, bottRightY)
        resize_ratio = min(self.input_size / org_w, self.input_size / org_h)
        dw = (self.input_size - resize_ratio * org_w) / 2  # padding width
        dh = (self.input_size - resize_ratio * org_h) / 2  # padding height
        pred_coor[:, [0, 2]] = (pred_coor[:, [0, 2]] - dw) / resize_ratio
        pred_coor[:, [1, 3]] = (pred_coor[:, [1, 3]] - dh) / resize_ratio

        # make sure topLeft and bottRight are both within the image
        pred_coor = np.c_[np.maximum(pred_coor[:, :2], [0, 0]),
                          np.minimum(pred_coor[:, 2:], [org_w - 1, org_h - 1])]

        # if topLeftX < bottRightX or topLeftY < bottRightY, the boxes are valid
        valid_mask = np.logical_and(pred_coor[:, 0] < pred_coor[:, 2],
                                    pred_coor[:, 1] < pred_coor[:, 3])
        pred_coor = pred_coor[valid_mask, :]
        confs = confs[valid_mask]
        class_probs = class_probs[valid_mask, :]

        # find classes for each boxes. Scale the scores of the classes by confidences
        classes = np.argmax(class_probs, axis=1)
        scores = confs * class_probs[np.arange(len(classes)), classes]

        # only keep boxes with scores above threshold
        score_mask = scores > self.SCORE_THRESHOLD
        coors = pred_coor[score_mask, :]
        scores = scores[score_mask]
        classes = classes[score_mask]
        return np.c_[coors, scores, classes]

    def _nms(self, pred_bboxes):
        classes_in_img = list(set(pred_bboxes[:, 5]))
        best_bboxes = []

        for cls in classes_in_img:
            cls_mask = (pred_bboxes[:, 5] == cls)
            cls_bboxes = pred_bboxes[cls_mask]

            while len(cls_bboxes) > 0:
                max_ind = np.argmax(cls_bboxes[:, 4])
                best_bbox = cls_bboxes[max_ind]
                best_bboxes.append(best_bbox)
                cls_bboxes = np.r_[cls_bboxes[: max_ind], cls_bboxes[max_ind + 1:]]
                iou = self._bboxes_iou(best_bbox[np.newaxis, :4], cls_bboxes[:, :4])
                weight = np.ones((len(iou),), dtype=np.float32)

                iou_mask = iou > self.IOU_THRESHOLD
                weight[iou_mask] = 0.0

                cls_bboxes[:, 4] = cls_bboxes[:, 4] * weight
                score_mask = cls_bboxes[:, 4] > 0.
                cls_bboxes = cls_bboxes[score_mask]

        return np.array(best_bboxes)

    @staticmethod
    def _bboxes_iou(boxes1, boxes2):
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

    def _preporcess(self, img):
        img = img.copy().astype(np.float32)

        # the longer side must become self.input_size
        ih, iw = self.input_size, self.input_size
        h,  w, _ = img.shape
        scale = min(iw/w, ih/h)
        nw, nh = int(scale * w), int(scale * h)
        image_resized = cv2.resize(img, (nw, nh))

        # pad the image equally on all sides, ie, center the resized original image on padded image
        image_paded = np.full(shape=[ih, iw, 3], fill_value=128.0)
        dw, dh = (iw - nw) // 2, (ih - nh) // 2
        image_paded[dh:nh+dh, dw:nw+dw, :] = image_resized

        image_paded = image_paded / 255.0  # all BRG values are normalized to [0.0, 1.0]

        # image_padded used to be (416, 416, 3). After next statement, it is (1, 416, 416, 3)
        image_paded = image_paded[np.newaxis, ...]  # add one more dimension at the beginning
        return image_paded
