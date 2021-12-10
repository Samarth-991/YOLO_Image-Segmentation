import cv2
import os
import numpy as np
import json
import onnxruntime
from app.tool.darknet2onnx import transform_to_onnx


class YOLO2ONXX:
    def __init__(self, config_path, model_cfg):
        self.config = config_path
        self.yolo_cfg = model_cfg.replace('.cfg', '_custom.cfg')
        with open(self.config, 'r') as f:
            self.parameters = json.load(f)
        # define weights and Yolo cfg
        self.weights_dir = os.path.join(self.parameters['DATA_PATH'],
                                        os.path.join(self.parameters['MODEL_DIR'], 'backup'))
        self.yolo_weights = os.path.join(self.weights_dir, 'instance_segment_custom_188000.weights')
        print(self.yolo_weights,self.yolo_cfg)
        self.transform()
        self.fit()


    def transform(self):
        # Transform to onnx
        self.onnx_model = transform_to_onnx(self.yolo_cfg, self.yolo_weights, 1)
        session = onnxruntime.InferenceSession(self.onnx_model)
        print("ONXX model loaded successfully")
        print("The model expects input shape: ", session.get_inputs()[0].shape)

    def fit(self):
        pass
