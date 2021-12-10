from sklearn.pipeline import Pipeline
from app import data_curator
from app import agument_dataset
from app import train_yolosegmentation
from app import run_inference


CONFIG_PATH = "app/cfg/parameter.json"
MODEL_CONFIG = "app/cfg/yolo_segment.cfg"
EVAL_CONFIG = "app/cfg/eval.json"
pipe = Pipeline(
    [
        ('Data_Creation', data_curator.Data_Creation(config_path=CONFIG_PATH)),
        ('Data_Augmentation', agument_dataset.Data_Augmentation(config_path=CONFIG_PATH,augment_data=False)),
        ('Train_YOLO_ImageSegmentation',train_yolosegmentation.Train_YOLO_ImageSegmentation(config_path=CONFIG_PATH, model_cfg=MODEL_CONFIG)),
        ('Run_inference', run_inference.Run_inference(eval_cfg=EVAL_CONFIG, yolo_cfg=MODEL_CONFIG))
    ]
)
