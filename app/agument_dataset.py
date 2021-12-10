import os
import numpy as np
import imageio
import json
import logging
import cv2
from app.utils import utils
import albumentations as A

logger = logging.getLogger(__name__)


class Data_Augmentation:
    def __init__(self, config_path, augment_data=False):
        self.Config_path = config_path
        with open(self.Config_path, 'r') as f:
            self.parameters = json.load(f)
        self.perform_augmentation = augment_data
        self.samples2augment = self.parameters['SAMPLES2AUGMENT']
        self.data_transform = A.Compose([A.HorizontalFlip(p=1)])
        self.train_path = os.path.join(self.parameters['DATA_PATH'],
                                       os.path.join(self.parameters['MODEL_DIR'], 'train/images'))
        # call transform and fit functions
        self.transform()
        self.fit()

    def read_image(self, image_path, mask=False):
        if mask:
            image = cv2.imread(image_path)
        else:
            image = cv2.imread(image_path)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        return image

    def transform(self):
        self.random_samples = [os.path.join(self.train_path, image_name) for image_name in
                               np.random.choice(os.listdir(self.train_path), self.samples2augment)]

    def fit(self):
        image_transformation = A.Compose([
            A.HorizontalFlip(p=1),
            A.RandomBrightnessContrast(p=0.2),
        ])
        if self.perform_augmentation:
            logging.info("Augmenting {} samples randomly".format(self.samples2augment ))
            for img_path in self.random_samples:
                fname = os.path.basename(img_path).split('.')[0]
                rgb_img = self.read_image(img_path)
                mask_img = self.read_image(img_path.replace('images', 'mask'), mask=True)
                transformed = image_transformation(image=rgb_img, mask=mask_img)
                # create Augmnetations
                imageio.imwrite(
                    os.path.join(os.path.dirname(img_path), fname + '_augmented' + self.parameters['IMAGES_PREFIX']),
                    transformed['image'])
                imageio.imwrite(os.path.join(os.path.dirname(img_path).replace('images', 'mask'),
                                             fname + '_augmented' + self.parameters['MASK_PREFIX']), transformed['mask'])

            # re-write the train.txt file with changes
            self.train_files = [os.path.join(self.train_path, img_name) for img_name in os.listdir(self.train_path) if
                                img_name.endswith(self.parameters['IMAGES_PREFIX'])]

            obj_path = os.path.join(self.parameters['DATA_PATH'], os.path.join(self.parameters['MODEL_DIR'], 'obj'))
            utils.write_data(self.train_files, os.path.join(obj_path, 'train.txt'))
        else:
            print("NO AUGMENTATION..")
            logging.info("Augmentaion is Paused..")