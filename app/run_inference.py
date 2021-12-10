import os
import cv2
import numpy as np
import json
from app.utils import utils
from dotenv import load_dotenv

load_dotenv()
from app.utils import darknet


class Run_inference:
    def __init__(self, eval_cfg, yolo_cfg, validataionpath='val/images'):
        print(os.getenv('DARKNET_PATH'))
        # read evaluation config
        self.eval_config = eval_cfg
        with open(self.eval_config, 'r') as f:
            self.parameters = json.load(f)
        # yolo model config
        self.yolo_cfg = yolo_cfg.replace('.cfg', '_custom.cfg')
        # Model weights path
        self.model_path = os.path.join(self.parameters['DATA_PATH'], self.parameters['MODEL_DIR'])
        self.weights_path = os.path.join(self.parameters['DATA_PATH'],
                                         os.path.join(self.parameters['MODEL_DIR'], self.parameters['WEIGHTS_DIR']))
        self.weights = os.path.join(self.weights_path, self.parameters["MODEL_NAME"])
        # meta data path
        self.meta_data = os.path.join(self.model_path, 'obj.data')
        self.testimgs = list()
        self.transform()
        self.fit()

    def transform(self):

        self.results_path = self.parameters['RESULT_PATH']
        self.test_path = self.parameters['TEST_PATH']
        if os.path.isdir(self.test_path):
            self.testimgs = [os.path.join(self.test_path, img_name) for img_name in os.listdir(self.test_path) if
                             img_name.endswith(self.parameters['IMAGE_PREFIX'])]

        elif os.path.isfile(self.test_path):
            self.testimgs = list(self.test_path)

    def fit(self):
        net,cls_names,c_map = darknet.load_network(self.yolo_cfg,self.weights,self.meta_data)
        for fname in self.testimgs:
            darknet.predict_image_mask(fname,net)
            mask,overlay = utils.visulize_colormap(fname,'test.png',c_map,len(cls_names))
            out_path = os.path.join(self.results_path,os.path.basename(fname))
            cv2.imwrite(out_path,overlay)

