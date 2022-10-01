import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
from tensorflow.python.client import device_lib
print(device_lib.list_local_devices())
import shutil
import tensorflow as tf
from Yolov3.dataset import Dataset
from Yolov3.yolov3 import Create_Yolo, compute_loss
from Yolov3.config import *
from mAP_evaluation import get_mAP
from Yolov3.utils import load_yolo_weights

def main_train_original():

    gpus = tf.config.experimental.list_physical_devices('GPU')
    print(f'GPUs {gpus}')
    if len(gpus) > 0:
        try:
            tf.config.experimental.set_memory_growth(gpus[0], True)
        except RuntimeError:
            pass

    if os.path.exists(TRAIN_LOGDIR_ORIGINAL): shutil.rmtree(TRAIN_LOGDIR_ORIGINAL)
    writer = tf.summary.create_file_writer(TRAIN_LOGDIR_ORIGINAL)

    trainset = Dataset('train', TRAIN_INPUT_SIZE_ORIGINAL, 0)
    testset = Dataset('test', TRAIN_INPUT_SIZE_ORIGINAL, 1)

    steps_per_epoch = len(trainset)
    global_steps = tf.Variable(-1, trainable=False, dtype=tf.int64)
    Darknet = Create_Yolo(input_size=TRAIN_INPUT_SIZE_ORIGINAL, CLASSES=YOLO_COCO_CLASSES)
    load_yolo_weights(Darknet, YOLO_V3_WEIGHTS) # use darknet weights test improvment

    yolo = Create_Yolo(input_size=TRAIN_INPUT_SIZE_ORIGINAL, training=True, CLASSES=TRAIN_CLASSES)

    for i, l in enumerate(Darknet.layers):
        layer_weights = l.get_weights()
        if layer_weights != []:
            try:
                yolo.layers[i].set_weights(layer_weights)
            except:
                print("skipping", yolo.layers[i].name)

    optimizer = tf.keras.optimizers.Adam()

    def train_step(image_data, target):
        with tf.GradientTape() as tape:
            pred_result = yolo(image_data, training=True)
            giou_loss = conf_loss = prob_loss = 0

            # optimizing process
            grid = 3
            for i in range(grid):
                conv, pred = pred_result[i * 2], pred_result[i * 2 + 1]
                loss_items = compute_loss(pred, conv, *target[i], i, CLASSES=TRAIN_CLASSES)
                giou_loss += loss_items[0]
                conf_loss += loss_items[1]
                prob_loss += loss_items[2]

            total_loss = giou_loss + conf_loss + prob_loss

            gradients = tape.gradient(total_loss, yolo.trainable_variables)
            optimizer.apply_gradients(zip(gradients, yolo.trainable_variables))
            global_steps.assign_add(1)

            optimizer.lr.assign(TRAIN_LR_END_ORIGINAL)
            # writing summary data
            with writer.as_default():
                tf.summary.scalar("lr", optimizer.lr, step=global_steps)
                tf.summary.scalar("loss/total_loss", total_loss, step=global_steps)
                tf.summary.scalar("loss/giou_loss", giou_loss, step=global_steps)
                tf.summary.scalar("loss/conf_loss", conf_loss, step=global_steps)
                tf.summary.scalar("loss/prob_loss", prob_loss, step=global_steps)
            writer.flush()

        return global_steps.numpy(), optimizer.lr.numpy(), giou_loss.numpy(), conf_loss.numpy(), prob_loss.numpy(), total_loss.numpy()

    validate_writer = tf.summary.create_file_writer(TRAIN_LOGDIR_ORIGINAL)

    def validate_step(image_data, target):
        with tf.GradientTape() as tape:
            pred_result = yolo(image_data, training=False)
            giou_loss = conf_loss = prob_loss = 0

            # optimizing process
            grid = 3
            for i in range(grid):
                conv, pred = pred_result[i * 2], pred_result[i * 2 + 1]
                loss_items = compute_loss(pred, conv, *target[i], i, CLASSES=TRAIN_CLASSES)
                giou_loss += loss_items[0]
                conf_loss += loss_items[1]
                prob_loss += loss_items[2]

            total_loss = giou_loss + conf_loss + prob_loss

        return giou_loss.numpy(), conf_loss.numpy(), prob_loss.numpy(), total_loss.numpy()

    mAP_model = Create_Yolo(input_size=YOLO_INPUT_SIZE_ORIGINAL, CLASSES=TRAIN_CLASSES)  # create second model to measure mAP

    counter = 0
    for epoch in range(TRAIN_EPOCHS_ORIGINAL):
        for image_data, target in trainset:
            counter += 1
            results = train_step(image_data, target)
            cur_step = results[0] % steps_per_epoch
            print(
                "epoch:{:2.0f} step:{:5.0f}/{}, lr:{:.6f}, giou_loss:{:7.2f}, conf_loss:{:7.2f}, prob_loss:{:7.2f}, total_loss:{:7.2f}"
                    .format(epoch, cur_step, steps_per_epoch, results[1], results[2], results[3], results[4],
                            results[5]))

        count, giou_val, conf_val, prob_val, total_val = 0., 0, 0, 0, 0
        for image_data, target in testset:
            results = validate_step(image_data, target)
            count += 1
            giou_val += results[0]
            conf_val += results[1]
            prob_val += results[2]
            total_val += results[3]
        # writing validate summary data
        with validate_writer.as_default():
            tf.summary.scalar("validate_loss/total_val", total_val / count, step=epoch)
            tf.summary.scalar("validate_loss/giou_val", giou_val / count, step=epoch)
            tf.summary.scalar("validate_loss/conf_val", conf_val / count, step=epoch)
            tf.summary.scalar("validate_loss/prob_val", prob_val / count, step=epoch)
        validate_writer.flush()

        print("\n\ngiou_val_loss:{:7.2f}, conf_val_loss:{:7.2f}, prob_val_loss:{:7.2f}, total_val_loss:{:7.2f}\n\n".
              format(giou_val / count, conf_val / count, prob_val / count, total_val / count))

        save_directory = os.path.join(TRAIN_ORIGINAL_END_FOLDER, TRAIN_MODEL_ORIGINAL_NAME)
        yolo.save_weights(save_directory)

    # measure mAP of trained model
    try:
        mAP_model.load_weights(save_directory)  # use keras weights
        get_mAP(mAP_model, testset, TEST_INPUT_SIZE_ORIGINAL, score_threshold=TEST_SCORE_THRESHOLD, iou_threshold=TEST_IOU_THRESHOLD)
    except UnboundLocalError:
        print(
            "You don't have saved model weights to measure mAP, check TRAIN_SAVE_BEST_ONLY and TRAIN_SAVE_CHECKPOINT lines in configs.py")


if __name__ == '__main__':
    main_train_original()
