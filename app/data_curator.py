import os
import numpy as np
import splitfolders
import json
import logging
from app.utils import utils

logger = logging.getLogger(__name__)


class Data_Creation:
    def __init__(self, config_path, verbose=1):
        self.Config_path = config_path
        self.parameters = None
        # define variables
        self.train_files = list()
        self.valid_files = list()
        self.transform()
        self.fit()

    def transform(self):
        if not os.path.isfile(self.Config_path):
            raise ("Configration path unresolved.")
        with open(self.Config_path, 'r') as f:
            self.parameters = json.load(f)
        # define model path
        self.arena_Path = os.path.join(self.parameters['DATA_PATH'],  self.parameters['MODEL_DIR'])

        if not os.path.isdir(self.arena_Path):
            splitfolders.ratio(os.path.join(self.parameters['DATA_PATH'], self.parameters['RAW_DIR']),
                               output=self.arena_Path, seed=1337,
                               ratio=(self.parameters['TRAIN_RATIO'], self.parameters['VALID_RATIO']),
                               group_prefix=None)

        train_path = os.path.join(self.arena_Path, 'train/images')
        validation_path = os.path.join(self.arena_Path, 'val/images')

        self.train_files = [os.path.join(train_path, img_name) for img_name in os.listdir(train_path) if
                            img_name.endswith(self.parameters['IMAGES_PREFIX'])]
        self.valid_files = [os.path.join(validation_path, img_name) for img_name in os.listdir(validation_path) if
                            img_name.endswith(self.parameters['IMAGES_PREFIX'])]

    def fit(self):
        yoloObj_path = os.path.join(self.arena_Path, 'obj')
        if not os.path.isdir(yoloObj_path):
            utils.make_dir(yoloObj_path)
        # write data in train and val
        utils.write_data(self.train_files, os.path.join(yoloObj_path, 'train.txt'))
        utils.write_data(self.valid_files, os.path.join(yoloObj_path, 'val.txt'))
        # make backup and results path
        utils.make_dir(os.path.join(self.arena_Path, 'backup'))
        utils.make_dir(os.path.join(self.arena_Path, 'results'))
        # create obj-data
        utils.write_objdata(obj_file=os.path.join(self.arena_Path, 'obj.data'),
                            n_classes=self.parameters['N_CLASSES'],
                            train_file=os.path.join(yoloObj_path, 'train.txt'),
                            val_file=os.path.join(yoloObj_path, 'val.txt'),
                            names_file=os.path.join(yoloObj_path, 'obj.names'),
                            result_path=os.path.join(self.arena_Path, 'result'),
                            backup=os.path.join(self.arena_Path, 'backup')
                            )

