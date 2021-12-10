import os
import logging
import json
import logging
from app.utils import utils

logger = logging.getLogger(__name__)


class Train_YOLO_ImageSegmentation:
    def __init__(self, config_path, model_cfg):
        self.Model_cfg = model_cfg
        self.Config_path = config_path
        with open(self.Config_path, 'r') as f:
            self.parameters = json.load(f)
        self.obj_data = os.path.join(self.parameters['DATA_PATH'],
                                     os.path.join(self.parameters['MODEL_DIR'], 'obj.data'))
        self.yolo_path = self.parameters['YOLO_DIR']
        self.pretrained_path = os.path.join(self.yolo_path, self.parameters['pre_trained_weights'])

        self.segment_config = None
        self.transform()
        self.fit()

    def transform(self):
        self.segment_config = utils.parse_config_file(self.Model_cfg, nb_classes=self.parameters['N_CLASSES'],
                                                      network_size=(
                                                          self.parameters['IMG_WIDTH'], self.parameters['IMG_HEIGHT']),
                                                      epochs=self.parameters['EPOCHS'],
                                                      batch_size=self.parameters['BATCH_SIZE'],
                                                      learning_rate=self.parameters['LEARNING_RATE']
                                                      )

    def fit(self):
        train_cmd = os.path.join(self.yolo_path, 'darknet') \
                    + ' segmenter train ' + self.obj_data + ' ' + self.segment_config + \
                    ' ' + self.pretrained_path
        os.system(train_cmd)
