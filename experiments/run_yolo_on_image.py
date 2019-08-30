import os
import tkinter
from tkinter.filedialog import askopenfilename

import cv2

from tf_yolov3.process_image import YOLOV3
from tf_yolov3.utils import box_objects, show_images_in_windows, milliseconds

file_dir = os.path.dirname(__file__)


if __name__ == '__main__':
    t0 = milliseconds()
    yolo = YOLOV3()
    t1 = milliseconds()
    print(f'init time: {t1-t0} ms')

    tkinter.Tk().withdraw()
    while 1:
        image_path = askopenfilename(initialdir="D:/projects/images/objects")
        if image_path is None:
            break

        original_image = cv2.imread(image_path)
        if original_image is None:
            break

        t0 = milliseconds()
        bboxes, labels = yolo.process(original_image)
        t1 = milliseconds()
        print(f'{t1-t0} ms elapsed')
        img = box_objects(original_image, bboxes, labels)
        show_images_in_windows([img], ['Test YOLO'], (img.shape[1], img.shape[0]))
