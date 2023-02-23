###################################################################################################
# Title: Model evaluator
# Description: Evaluate trained model and print out results.
# 
# Jure Rebernik magistrska naloga
###################################################################################################

import os
import argparse
import tensorflow as tf
from object_detection.utils import config_util
from object_detection.protos import pipeline_pb2
from google.protobuf import text_format

CUSTOM_MODEL_NAME = 'plate_model_320'
PRETRAINED_MODEL_NAME = 'ssd_mobilenet_v2_fpnlite_320x320_coco17_tpu-8'
PRETRAINED_MODEL_URL = 'http://download.tensorflow.org/models/object_detection/tf2/20200711/ssd_mobilenet_v2_fpnlite_320x320_coco17_tpu-8.tar.gz'
TF_RECORD_SCRIPT_NAME = 'generate_tfrecord.py'
LABEL_MAP_NAME = 'label_map.pbtxt'
LABELS = [{'name':'plate', 'id':1}]

# TODO: Make this in a seperate folder and use include to include paths. Do the same for every file.
paths = {
    'WORKSPACE_PATH': os.path.join('tensorflow', 'workspace'),
    'SCRIPTS_PATH': os.path.join('tensorflow','scripts'),
    'APIMODEL_PATH': os.path.join('tensorflow','models'),
    'ANNOTATION_PATH': os.path.join('tensorflow', 'workspace','annotations'),
    'IMAGE_PATH': os.path.join('dataset', 'images'),
    'MODEL_PATH': os.path.join('tensorflow', 'workspace','models'),
    'PRETRAINED_MODEL_PATH': os.path.join('tensorflow', 'workspace','pre-trained-models'),
    'CHECKPOINT_PATH': os.path.join('tensorflow', 'workspace','models',CUSTOM_MODEL_NAME), 
    'OUTPUT_PATH': os.path.join('tensorflow', 'workspace','models',CUSTOM_MODEL_NAME, 'export'), 
    'TFJS_PATH':os.path.join('tensorflow', 'workspace','models',CUSTOM_MODEL_NAME, 'tfjsexport'), 
    'TFLITE_PATH':os.path.join('tensorflow', 'workspace','models',CUSTOM_MODEL_NAME, 'tfliteexport'), 
    'PROTOC_PATH':os.path.join('tensorflow','protoc')
 }

files = {
    'PIPELINE_CONFIG':os.path.join('tensorflow', 'workspace','models', CUSTOM_MODEL_NAME, 'pipeline.config'),
    'TF_RECORD_SCRIPT': os.path.join(paths['SCRIPTS_PATH'], TF_RECORD_SCRIPT_NAME), 
    'LABELMAP': os.path.join(paths['ANNOTATION_PATH'], LABEL_MAP_NAME)
}


# Initiate argument parser
parser = argparse.ArgumentParser(
    description="Automatic model training script.")
parser.add_argument("-n",
                    "--num_steps",
                    help="Number of training steps.",
                    type=int)
# parser.add_argument("-l",
#                     "--labels_path",
#                     help="Path to the labels (.pbtxt) file.", type=str)

args = parser.parse_args()



if __name__ == '__main__':

    TRAINING_SCRIPT = os.path.join(paths['APIMODEL_PATH'], 'research', 'object_detection', 'model_main_tf2.py')
    cmd = "python {} --model_dir={} --pipeline_config_path={} --checkpoint_dir={}".format(TRAINING_SCRIPT, paths['CHECKPOINT_PATH'],files['PIPELINE_CONFIG'], paths['CHECKPOINT_PATH'])
    print(cmd) # python tensorflow/models/research/object_detection/model_main_tf2.py --model_dir=tensorflow/workspace/models/plate_model_320 --pipeline_config_path=tensorflow/workspace/models/plate_model_320/pipeline.config --num_train_steps=2000
    os.system(cmd)