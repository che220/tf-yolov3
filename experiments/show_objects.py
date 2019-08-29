import os
import numpy as np
import cv2

from tf_yolov3.utils import box_objects, show_images_in_windows

file_dir = os.path.dirname(__file__)


if __name__ == '__main__':
    bboxes = np.array([[484.85, 379.94, 541.93, 426.09, 0.7654, 2.0],
                       [606.25, 389.89, 622.31, 404.20, 0.6123, 2.0],
                       [498.04, 337.56, 558.32, 419.81, 0.5374, 5.0]])
    labels = np.array(['car', 'car', 'bus'])
    image_path = os.path.join(file_dir, "308.jpg")
    original_image = cv2.imread(image_path)
    img = box_objects(original_image, bboxes, labels)
    show_images_in_windows([img], ['Test show_objects()'], (img.shape[1], img.shape[0]))
