###################################################################################################
# Title: Freeze the trained model and convert to TFLite
# Description: Freeze/save the trained model from the latest checkpoint and convert to TFLite.
# 
# Jure Rebernik magistrska naloga
###################################################################################################

import os

FREEZE_SCRIPT_PATH = 'tensorflow/models/research/object_detection/exporter_main_v2.py'
CHECKPOINT_PATH = 'tensorflow/workspace/models/plate_model_320/check'
PIPELINE_CONFIG_PATH = 'tensorflow/workspace/models/plate_model_320/pipeline.config'
OUTPUT_PATH = 'tensorflow/workspace/models/plate_model_320/export'

TFLITE_SCRIPT = 'tensorflow/models/research/object_detection/export_tflite_graph_tf2.py'
TF_LITE_EXPORT_PATH = 'tensorflow/workspace/models/plate_model_320/export/tflite'

FROZEN_TFLITE_PATH = 'tensorflow/workspace/models/plate_model_320/export/tflite/saved_model'
TFLITE_MODEL = 'tensorflow/workspace/models/plate_model_320/export/tflite/saved_model/detect.tflite'

cmd = "python {} --input_type=image_tensor --pipeline_config_path={} --trained_checkpoint_dir={} --output_directory={}".format(FREEZE_SCRIPT_PATH, PIPELINE_CONFIG_PATH, CHECKPOINT_PATH, OUTPUT_PATH)
os.system(cmd)

cmd = "python {} --pipeline_config_path={} --trained_checkpoint_dir={} --output_directory={}".format(TFLITE_SCRIPT, PIPELINE_CONFIG_PATH, CHECKPOINT_PATH, TF_LITE_EXPORT_PATH)
os.system(cmd)

cmd = "tflite_convert \
--saved_model_dir={} \
--output_file={} \
--input_shapes=1,300,300,3 \
--input_arrays=normalized_input_image_tensor \
--output_arrays='TFLite_Detection_PostProcess','TFLite_Detection_PostProcess:1','TFLite_Detection_PostProcess:2','TFLite_Detection_PostProcess:3' \
--inference_type=FLOAT \
--allow_custom_ops".format(FROZEN_TFLITE_PATH, TFLITE_MODEL)
os.system(cmd)
