import os
import tkinter
from time import sleep
from tkinter.filedialog import askopenfilename

import cv2

from tf_yolov3.process_image import YOLOV3
from tf_yolov3.utils import box_objects, milliseconds, show_images_in_windows

file_dir = os.path.dirname(__file__)


if __name__ == '__main__':
    tkinter.Tk().withdraw()
    while 1:
        video_path = askopenfilename(initialdir="D:/tmp/snapshots")
        if video_path is None:
            break

        video = cv2.VideoCapture(video_path)
        cv2.namedWindow(video_path)
        w, h = int(video.get(cv2.CAP_PROP_FRAME_WIDTH)), int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
        cv2.resizeWindow(video_path, w, h)
        cnt = 0
        while 1:
            ret, frame = video.read()
            if not ret:
                break

            cnt += 1
            cv2.imshow(video_path, frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cv2.destroyWindow(video_path)
