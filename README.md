# Darknet_Image-Segmentation
Image-Segmentation task using Darknet-YOLOv4-Customized YOLO algorithm for image segmentation task on Road Surface Segmentation Dataset.The code contains Diluted convolutions which are part of DeelabV3 architecture, added in Darknet for Image segmentation.

[The Command to Run :Crude approach]
=========
Compile: 
	make -j8

Train: 

	./darknet segmenter train cfg/maskyolo.data cfg/instance_segment.cfg [pretrain weights attached4] 

Test:
	./darknet segmenter test cfg/maskyolo.data cfg/instance_segment.cfg [weights file] [image path]


[The Command to Run :Python API approach]
=========
Compile: 
	make -j8


Update Configrations:
	Update configrations file according to your dataset.
	parameter.json: Update paths Train image parameters and folder structure with model hyperparameters
	eval.json: Update evaluation config with evaluation metrics and performance
	yolo_segment: Yolo-segmentation config to change model layers and depth

class_names:
	You need to create obj.names file and place inside model_directory path for network to created multi-class image segmentation model

env_path:
	Add Darknet path in .env file for Darknet to train and evaluate model

Train Network:
	use train_pipeline.py to train the E2E pipeline.

Results:
	Masked results are placed in results directory

[Pretrain weights file and data]  
========  
1. https://www.dropbox.com/sh/9wrevnyzwfv8hg7/AAA1MJElri9aROsjaPTxO5KCa?dl=0
2. https://towardsdatascience.com/road-surface-semantic-segmentation-4d65b045245

[How to Train with Your Own Dataset ?]  
========  
1. Create Dataset:

	original RGB image (RGB image saved in the images folder in dataset)   	
	Mask image(pixel value is 0, 1, 2, 3 if you have 3 classes + background) to be saved in mask folder in dataset.
	Mask image needs to be downloaded as No-Color_mask data   

2. Download the Pre-trained weights:
	Pretrained weights path needs to be updated
	Toggle for Augmnetations if required in param configrations
			
3. Hyperparameter Tuning:
	Tune Hyper-parameters in params.config

4. Inference:
	Update eval config json for Inference on image or folder 

* Note: Script automatically selects images and mask paths as labels so no need to 



