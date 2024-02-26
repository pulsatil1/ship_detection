import os

# Model Parameters
BATCH_SIZE = 48
EDGE_CROP = 16
GAUSSIAN_NOISE = 0.1
UPSAMPLE_MODE = 'SIMPLE'
# downsampling inside the network
NET_SCALING = (1, 1)
# downsampling in preprocessing
IMG_SCALING = (3, 3)
# number of validation images to use
VALID_IMG_COUNT = 900
# maximum number of steps_per_epoch in training
MAX_TRAIN_STEPS = 10
AUGMENT_BRIGHTNESS = False
SAMPLES_PER_GROUP = 2000

BASE_DIR = '../data/airbus-ship-detection'
TRAIN_DIR = os.path.join(BASE_DIR, 'train_v2')
TEST_DIR = os.path.join(BASE_DIR, 'test_v2')
