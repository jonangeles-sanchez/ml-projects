"""The main of the Image Segmentation with Machine Learning
-----------------------------

About this Module
------------------
The goal of this module is initiate the run of the Image Segmentation with
Machine Learning and define core components..
"""

__author__ = "Benoit Lapointe"
__date__ = "2021-06-01"
__copyright__ = "Copyright 2021, labesoft"
__version__ = "1.0.0"

import argparse
import json
import os.path
from pathlib import Path

import matplotlib.pyplot as plt
import mrcnn.model as mrcnnmodel
import skimage.io
import tensorflow as tf
from mrcnn import visualize

import lib.coco as coco

BASE_DIR = Path(os.path.dirname(__file__))
MODEL_DIR = BASE_DIR.joinpath('mask_rcnn_coco.hy')
MODEL_FILE = BASE_DIR.joinpath('mask_rcnn_coco.h5')
CLASSES_FILE = BASE_DIR.joinpath('coco_classnames.json')
IMAGE_FILE = BASE_DIR.joinpath('office-2539844_1920.jpg')


class InferenceConfig(coco.CocoConfig):
    """The inference configuration limiting memory and gpu uses

    Setting batch size equal to 1 since we'll be running inference on one
    image at a time. Batch size = GPU_COUNT * IMAGES_PER_GPU
    """
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        # Restrict TensorFlow to only allocate 1GB of memory on the first GPU
        try:
            tf.config.experimental.set_virtual_device_configuration(
                gpus[0],
                [tf.config.experimental.VirtualDeviceConfiguration(
                    memory_limit=512 + 256)])
            logical_gpus = tf.config.experimental.list_logical_devices('GPU')
            print(len(gpus), "Physical GPUs,", len(logical_gpus),
                  "Logical GPUs")
        except RuntimeError as e:
            # Virtual devices must be set before GPUs have been initialized
            print(e)


if __name__ == '__main__':
    """Main entry point of the imsegment package"""
    # Parsing arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--image', default=str(IMAGE_FILE))
    args = parser.parse_args()

    # Create model objects in inference mode.
    config = InferenceConfig()
    config.display()
    model = mrcnnmodel.MaskRCNN(
        mode="inference", model_dir=str(MODEL_DIR), config=config
    )
    model.load_weights(str(MODEL_FILE), by_name=True)

    # Image to segment
    plt.figure(figsize=(12, 10))
    image = skimage.io.imread(args.image)
    skimage.io.imshow(image)

    # Run detection
    results = model.detect([image], verbose=1)

    # Visualize results
    r = results[0]
    with open(str(CLASSES_FILE)) as f:
        class_names = {int(k): v for k, v in json.load(f).items()}
    visualize.display_instances(
        image, r['rois'], r['masks'], r['class_ids'], class_names=class_names
    )
    mask = r['masks']
    mask = mask.astype(int)
    print(mask.shape)
