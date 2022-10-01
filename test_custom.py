import os

os.environ['CUDA_VISIBLE_DEVICES'] = '0'
from Yolov3.utils import detect_video, detect_image, Load_Yolo_model
from Yolov3.config import *


def main_test_custom():

    video_path = "./Data/swan.mp4"
    image_path = "./Data/testImage1.jpg"

    yolo = Load_Yolo_model(1)
    detect_image(yolo, image_path, "./IMAGES/plate_1_detect.jpg", input_size=YOLO_INPUT_SIZE_CUSTOM, show=True, CLASSES=TRAIN_CLASSES, rectangle_colors=(255,0,0))
    #detect_video(yolo, video_path, './Output/Custom/detected.mp4', input_size=YOLO_INPUT_SIZE_CUSTOM, show=True, CLASSES=TRAIN_CLASSES, rectangle_colors=(0, 0, 255))


if __name__ == '__main__':
    main_test_custom()
