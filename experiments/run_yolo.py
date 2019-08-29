import os
import cv2

from tf_yolov3.process_image import YOLOV3
from tf_yolov3.utils import show_objects

file_dir = os.path.dirname(__file__)

if __name__ == '__main__':
    image_path = os.path.join(file_dir, "308.jpg")
    original_image = cv2.imread(image_path)

    yolo = YOLOV3()
    bboxes, labels = yolo.process(original_image)
    show_objects(original_image, bboxes, labels, 'Test YOLO')
