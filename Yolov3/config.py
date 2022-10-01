
# YOLO options
YOLO_TYPE                   = "yolov3"
YOLO_FRAMEWORK              = "tf" # "tf" or "trt"
YOLO_V3_WEIGHTS             = "Yolov3/yolov3.weights"
YOLO_TRT_QUANTIZE_MODE      = "INT8" #INT8, FP16, FP32
YOLO_CUSTOM_WEIGHTS         = True # "checkpoints/yolov3_custom" # used in evaluate_mAP.py and custom model detection, if not using leave False
YOLO_COCO_CLASSES           = "Model_data/coco/coco.names"
YOLO_STRIDES                = [8, 16, 32]
YOLO_IOU_LOSS_THRESH        = 0.5
YOLO_ANCHOR_PER_SCALE       = 3
YOLO_MAX_BBOX_PER_SCALE     = 100
YOLO_INPUT_SIZE_ORIGINAL    = 224
YOLO_INPUT_SIZE_CUSTOM      = 224#480  #biger input size for better prediction
if YOLO_TYPE                == "yolov3":
    YOLO_ANCHORS            = [[[10,  13], [16,   30], [33,   23]],
                               [[30,  61], [62,   45], [59,  119]],
                               [[116, 90], [156, 198], [373, 326]]]

# Train options
TRAIN_YOLO_TINY             = False
TRAIN_SAVE_BEST_ONLY        = True # saves only best model according validation loss
TRAIN_SAVE_CHECKPOINT       = False # saves all best validated checkpoints in training process (may require a lot disk space)
TRAIN_CLASSES               = "Model_data/swan_names.txt"
TRAIN_ANNOT_PATH            = "Model_data/swan_train.txt"
TRAIN_LOGDIR_ORIGINAL       = "log_o"
TRAIN_LOGDIR_CUSTOM         = "log_c"
TRAIN_CHECKPOINTS_FOLDER_C  = "CustomModel/Checkpoints/Custom"
TRAIN_CHECKPOINTS_FOLDER_O  = "Checkpoints/Original"
TRAIN_ORIGINAL_END_FOLDER   = "OriginalModel"
TRAIN_CUSTOM_END_FOLDER     = "CustomModel"
TRAIN_MODEL_ORIGINAL_NAME   = f"{YOLO_TYPE}_original"
TRAIN_MODEL_CUSTOM_NAME     = f"{YOLO_TYPE}_custom"
TRAIN_LOAD_IMAGES_TO_RAM    = True # With True faster training, but need more RAM
TRAIN_BATCH_SIZE            = 1
TRAIN_INPUT_SIZE_ORIGINAL   = 224
TRAIN_INPUT_SIZE_CUSTOM     = 224
TRAIN_FROM_CHECKPOINT       = False # "checkpoints/yolov3_custom"
TRAIN_LR_INIT_CUSTOM        = 1e-4
TRAIN_LR_END_CUSTOM         = 1e-5
TRAIN_LR_END_ORIGINAL       = 1e-5
TRAIN_WARMUP_EPOCHS         = 1  #origin 0
TRAIN_EPOCHS_ORIGINAL       = 2  #origin 5
TRAIN_EPOCHS_CUSTOM         = 2  #origin 5

# TEST options
TEST_ANNOT_PATH             = "Model_data/swan_test.txt"
TEST_BATCH_SIZE             = 1
TEST_INPUT_SIZE_ORIGINAL    = 224
TEST_INPUT_SIZE_CUSTOM      = 224
TEST_DATA_AUG               = True
TEST_DECTECTED_IMAGE_PATH   = ""
TEST_SCORE_THRESHOLD        = 0.3
TEST_IOU_THRESHOLD          = 0.45