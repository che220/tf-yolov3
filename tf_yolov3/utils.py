import datetime as dt
import random
import cv2


def milliseconds():
    """

    Returns: milliseconds since epoch

    """
    return dt.datetime.now().timestamp()*1000.0


def show_images_in_windows(imgs, win_names, win_size):
    """
    display multiple images in multiple windows

    :param imgs:
    :param win_names:
    :param win_size:
    :return:
    """
    x = y = 0
    for i, img in enumerate(imgs):
        w_compress = img.shape[1] / win_size[0]
        h_compress = img.shape[0] / win_size[1]
        if w_compress > h_compress:
            w = win_size[0]
            h = img.shape[0] / w_compress
        else:
            w = img.shape[1] / h_compress
            h = win_size[1]
        w = int(w)
        h = int(h)

        win_name = win_names[i]
        cv2.namedWindow(win_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(win_name, w, h)
        cv2.moveWindow(win_name, x, y)
        cv2.imshow(win_name, img)
        x += w
    cv2.waitKey(0) & 0xFF  # for 64-bit machine
    cv2.destroyAllWindows()


def box_objects(img, boxes, labels):
    colors = {}
    for label in labels:
        colors[label] = (random.randint(128, 255),
                         random.randint(128, 255),
                         random.randint(128, 255))

    tmp_img = img.copy()
    if len(tmp_img.shape) == 2:
        tmp_img = cv2.cvtColor(tmp_img, cv2.COLOR_GRAY2BGR)

    for i, [x1, y1, x2, y2, conf, _] in enumerate(boxes):
        label = labels[i]
        c = colors[label]
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
        cv2.rectangle(tmp_img, (x1, y1), (x2, y2), color=c, thickness=1)

        msg = f'{label} {conf:.4f}'
        cv2.putText(tmp_img, msg, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color=c,
                    thickness=1)
    return tmp_img
