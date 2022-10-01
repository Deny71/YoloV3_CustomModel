import os
import cv2
import random
import numpy as np
from Yolov3.config import *


def parse_annotation(annotation, mAP='False'):
    if TRAIN_LOAD_IMAGES_TO_RAM:
        image_path = annotation[0]
        image = annotation[2]
    else:
        image_path = annotation[0]
        image = cv2.imread(image_path)

    bboxes = np.array([list(map(int, box.split(','))) for box in annotation[1]])


    image = noiseMaker(np.copy(image))
    image, bboxes = random_horizontal_flip(np.copy(image), np.copy(bboxes))
    image, bboxes = random_crop(np.copy(image), np.copy(bboxes))
    image, bboxes = random_translate(np.copy(image), np.copy(bboxes))

    return image, bboxes


def load_annotations(annot_path):
    final_annotations = []
    with open(annot_path, 'r') as f:
        txt = f.read().splitlines()
        annotations = [line.strip() for line in txt if len(line.strip().split()[1:]) != 0]
    #np.random.shuffle(annotations)

    for annotation in annotations:
        # fully parse annotations
        line = annotation.split()
        image_path, index = "", 1
        for i, one_line in enumerate(line):
            if not one_line.replace(",","").isnumeric():
                if image_path != "": image_path += " "
                image_path += one_line
            else:
                index = i
                break
        if not os.path.exists(image_path):
            raise KeyError("%s does not exist ... " %image_path)
        if TRAIN_LOAD_IMAGES_TO_RAM:
            image = cv2.imread(image_path)
        else:
            image = ''
        final_annotations.append([image_path, line[index:], image])
        cos = ""
        with open("D:\Dev\Python\Yolov3Comparison\Model_data\swan_test_noises_full.txt", 'w') as testList:
            for anno in final_annotations:
                imageOut, bboxesOut = parse_annotation(anno)
                cv2.imwrite(anno[0], imageOut)
                outStr = ""
                counter = 1

                for elem in np.ravel(bboxesOut):
                    if counter % 5 != 0:
                        outStr += str(elem) + ","
                    elif counter % 5 == 0 :
                        outStr += str(elem) + " "
                    counter += 1

                lineOut = anno[0] + " " + outStr + "\n"
                testList.write(lineOut)


def random_horizontal_flip(image, bboxes):
    if random.random() < 0.5:
        _, w, _ = image.shape
        image = image[:, ::-1, :]
        bboxes[:, [0,2]] = w - bboxes[:, [2,0]]

    return image, bboxes

def random_crop(image, bboxes):
    if random.random() < 0.5:
        h, w, _ = image.shape
        max_bbox = np.concatenate([np.min(bboxes[:, 0:2], axis=0), np.max(bboxes[:, 2:4], axis=0)], axis=-1)

        max_l_trans = max_bbox[0]
        max_u_trans = max_bbox[1]
        max_r_trans = w - max_bbox[2]
        max_d_trans = h - max_bbox[3]

        crop_xmin = max(0, int(max_bbox[0] - random.uniform(0, max_l_trans)))
        crop_ymin = max(0, int(max_bbox[1] - random.uniform(0, max_u_trans)))
        crop_xmax = max(w, int(max_bbox[2] + random.uniform(0, max_r_trans)))
        crop_ymax = max(h, int(max_bbox[3] + random.uniform(0, max_d_trans)))

        image = image[crop_ymin : crop_ymax, crop_xmin : crop_xmax]

        bboxes[:, [0, 2]] = bboxes[:, [0, 2]] - crop_xmin
        bboxes[:, [1, 3]] = bboxes[:, [1, 3]] - crop_ymin

    return image, bboxes

def random_translate(image, bboxes):
    if random.random() < 0.5:
        h, w, _ = image.shape
        max_bbox = np.concatenate([np.min(bboxes[:, 0:2], axis=0), np.max(bboxes[:, 2:4], axis=0)], axis=-1)

        max_l_trans = max_bbox[0]
        max_u_trans = max_bbox[1]
        max_r_trans = w - max_bbox[2]
        max_d_trans = h - max_bbox[3]

        tx = random.uniform(-(max_l_trans - 1), (max_r_trans - 1))
        ty = random.uniform(-(max_u_trans - 1), (max_d_trans - 1))

        M = np.array([[1, 0, tx], [0, 1, ty]])
        image = cv2.warpAffine(image, M, (w, h))

        bboxes[:, [0, 2]] = bboxes[:, [0, 2]] + tx
        bboxes[:, [1, 3]] = bboxes[:, [1, 3]] + ty

    return image, bboxes

def noiseMaker(image):

    noise_type = random.randint(1,4)

    if noise_type == 1:
        row, col, ch = image.shape
        mean = 0
        var = 0.1
        sigma = var ** 0.5
        gauss = np.random.normal(mean, sigma, (row, col, ch))
        gauss = gauss.reshape(row, col, ch)
        noisy = image + gauss
        return noisy

    elif noise_type == 2:
        row, col, ch = image.shape
        s_vs_p = 0.5
        amount = 0.004
        out = np.copy(image)
        # Salt mode
        num_salt = np.ceil(amount * image.size * s_vs_p)
        coords = [np.random.randint(0, i - 1, int(num_salt))
                  for i in image.shape]
        out[coords] = 1

        # Pepper mode
        num_pepper = np.ceil(amount * image.size * (1. - s_vs_p))
        coords = [np.random.randint(0, i - 1, int(num_pepper))
                  for i in image.shape]
        out[coords] = 0
        return out

    elif noise_type == 3:
        vals = len(np.unique(image))
        vals = 2 ** np.ceil(np.log2(vals))
        noisy = np.random.poisson(image * vals) / float(vals)
        return noisy

    elif noise_type == 4:
        row, col, ch = image.shape
        gauss = np.random.randn(row, col, ch)
        gauss = gauss.reshape(row, col, ch)
        noisy = image + image * gauss
        return noisy


load_annotations('D:\Dev\Python\Yolov3Comparison\Model_data\swan_test.txt')
print("Noises applied correct!")
