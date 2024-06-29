#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @FileName  :read_av_sas.py
# @Time      :2024/5/29 13:49
# @Author    :Darcy
# @Desc      :
from ultralytics import YOLO
import supervision as sv
import cv2


def check_2():

    MODEL_PATH = 'models/yolov10x.pt'
    model = YOLO(MODEL_PATH)
    box_annotator = sv.BoxAnnotator()  # 提示警告

    # image = cv2.imread(IMAGE_PATH)
    def img_resize(image):
        height, width = image.shape[0], image.shape[1]
        # 设置新的图片分辨率框架 640x369 1280×720 1920×1080
        width_new = 1920
        height_new = 1080
        # 判断图片的长宽比率
        if width / height >= width_new / height_new:
            img_new = cv2.resize(image, (width_new, int(height * width_new / width)))
        else:
            img_new = cv2.resize(image, (int(width * height_new / height), height_new))
        return img_new

    def yolo_deal(image):
        results = model(source=image, conf=0.5, verbose=False)[0]
        detections = sv.Detections.from_ultralytics(results)
        category_dict = {
            0: 'person', 1: 'bicycle', 2: 'car', 3: 'motorcycle', 4: 'airplane', 5: 'bus',
            6: 'train', 7: 'truck', 8: 'boat', 9: 'traffic light', 10: 'fire hydrant',
            11: 'stop sign', 12: 'parking meter', 13: 'bench', 14: 'bird', 15: 'cat',
            16: 'dog', 17: 'horse', 18: 'sheep', 19: 'cow', 20: 'elephant', 21: 'bear',
            22: 'zebra', 23: 'giraffe', 24: 'backpack', 25: 'umbrella', 26: 'handbag',
            27: 'tie', 28: 'suitcase', 29: 'frisbee', 30: 'skis', 31: 'snowboard',
            32: 'sports ball', 33: 'kite', 34: 'baseball bat', 35: 'baseball glove',
            36: 'skateboard', 37: 'surfboard', 38: 'tennis racket', 39: 'bottle',
            40: 'wine glass', 41: 'cup', 42: 'fork', 43: 'knife', 44: 'spoon', 45: 'bowl',
            46: 'banana', 47: 'apple', 48: 'sandwich', 49: 'orange', 50: 'broccoli',
            51: 'carrot', 52: 'hot dog', 53: 'pizza', 54: 'donut', 55: 'cake',
            56: 'chair', 57: 'couch', 58: 'potted plant', 59: 'bed', 60: 'dining table',
            61: 'toilet', 62: 'tv', 63: 'laptop', 64: 'mouse', 65: 'remote', 66: 'keyboard',
            67: 'cell phone', 68: 'microwave', 69: 'oven', 70: 'toaster', 71: 'sink',
            72: 'refrigerator', 73: 'book', 74: 'clock', 75: 'vase', 76: 'scissors',
            77: 'teddy bear', 78: 'hair drier', 79: 'toothbrush'
        }
        print("detections--->", detections)
        labels = [
            f"{category_dict[class_id]} {confidence:.2f}"
            for class_id, confidence in zip(detections.class_id, detections.confidence)
        ]
        annotated_image = box_annotator.annotate(
            image.copy(), detections=detections, labels=labels
        )
        return annotated_image

    # 向共享缓冲栈中写入数据:
    cap = cv2.VideoCapture(0)
    # cap = cv2.VideoCapture('http://admin:admin@192.168.10.35:8081')
    # cap = cv2.VideoCapture('https://gbs.***.com:10010/sms/34020000002020000001/hls/34020000001180000187/live.m3u8?token=y0Xj9Ad74-XnmgpFB6sV8sYiefo2b')
    ret, frame = cap.read()
    print(cap.get(5), '<---------视频帧率')
    time_c = 0
    while ret:
        # 读取视频帧
        time_c += 1
        # 设置每 10 帧输出一次
        if (time_c % 1) != 0:
            ret, frame = cap.read()
            yolo_img = frame
        else:
            ret, frame = cap.read()
            yolo_img = yolo_deal(frame)
        if time_c > 1000:
            time_c = 0
        # 显示视频帧
        cv2.imshow("frame", img_resize(yolo_img))
        # 等候1ms,播放下一帧，或者按q键退出
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    # 等待按键然后关闭窗口
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == '__main__':
    check_2()
