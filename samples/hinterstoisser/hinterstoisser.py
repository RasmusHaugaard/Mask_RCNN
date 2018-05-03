import numpy as np
from pathlib import Path
import yaml
import os
import sys
import skimage.io

# Root directory of the project
ROOT_DIR = os.path.abspath("../../")

# Import Mask RCNN
sys.path.append(ROOT_DIR)  # To find local version of the library
from mrcnn.config import Config
from mrcnn import utils


class HinterstoisserConfig(Config):
    NAME = "hinterstoisser"

    GPU_COUNT = 1
    IMAGES_PER_GPU = 2

    NUM_CLASSES = 1 + 15  # background + 15 classes

    IMAGE_MIN_DIM = 480
    IMAGE_MAX_DIM = 640

    IMAGE_RESIZE_MODE = "square"  # 640x480

    # Use smaller anchors because our image and objects are small
    RPN_ANCHOR_SCALES = (16, 32, 64, 128, 256)  # anchor side in pixels

    # Aim to allow ROI sampling to pick 33% positive ROIs.
    TRAIN_ROIS_PER_IMAGE = 128

    STEPS_PER_EPOCH = 300
    VALIDATION_STEPS = 30

    DEPTH_MODE = "before_rpn"
    DEPTH_CHANNELS = 1


class HinterstoisserDataset(utils.Dataset):
    def load_hinterstoisser(self, hinterstoisserDir, scene_id, subset):
        self.add_class("hinterstoisser", 1, "ape")
        self.add_class("hinterstoisser", 2, "clamp")
        self.add_class("hinterstoisser", 3, "bowl")
        self.add_class("hinterstoisser", 4, "camera")
        self.add_class("hinterstoisser", 5, "can")
        self.add_class("hinterstoisser", 6, "cat")
        self.add_class("hinterstoisser", 7, "cup")
        self.add_class("hinterstoisser", 8, "driller")
        self.add_class("hinterstoisser", 9, "duck")
        self.add_class("hinterstoisser", 10, "box")
        self.add_class("hinterstoisser", 11, "glue")
        self.add_class("hinterstoisser", 12, "hole puncher")
        self.add_class("hinterstoisser", 13, "iron")
        self.add_class("hinterstoisser", 14, "lamp")
        self.add_class("hinterstoisser", 15, "phone")

        out_dir = Path(hinterstoisserDir + '/test/{:02}'.format(scene_id))
        print(out_dir)
        assert out_dir.exists()
        subset_path = Path(hinterstoisserDir + '/' + subset + '.yml')
        print(subset_path)
        assert subset_path.exists()
        with subset_path.open() as stream:
            image_ids = yaml.load(stream)[scene_id]

        for image_id in image_ids:
            self.add_image("hinterstoisser", image_id=image_id,
                           path=str(out_dir / 'rgb' / '{:04}.png'.format(image_id)))

    def image_reference(self, image_id):
        """Return the image_def data of the image."""
        info = self.image_info[image_id]
        if info["source"] == "hinterstoisser":
            return info["hinterstoisser"]
        else:
            super(self.__class__).image_reference(image_id)

    def load_image(self, image_id):
        info = self.image_info[image_id]
        rgb = skimage.io.imread(info['path']).astype(np.uint16)
        depth = skimage.io.imread(str(
            Path(info['path']).parent.parent / 'depth' / '{:04}.png'.format(info['id'])
        )).reshape((*rgb.shape[:2], 1))
        return np.concatenate((rgb, depth), axis=-1)

    def load_mask(self, image_id):
        info = self.image_info[image_id]
        mask = skimage.io.imread(str(
            Path(info['path']).parent.parent / 'mask' / '{:04}.png'.format(info['id'])
        ))

        masks = []
        class_ids = []

        for class_id in range(1, 16):
            m = mask == class_id
            area = m.sum()
            if area > 20 ** 2:
                masks.append(m)
                class_ids.append(class_id)

        masks = np.stack(masks, axis=-1)

        return masks, np.array(class_ids, dtype=np.int32)
