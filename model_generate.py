###################################################################################################
# Title: Model generator
# Description: Download model and generate necessary files.
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

    if args.num_steps is None:
        raise Exception("Error: num_steps argument not provided, exiting...")

    # Create label map
    cmd = "rm " + files['LABELMAP'] + " -r"
    os.system(cmd)

    with open(files['LABELMAP'], 'w') as f:
        for label in LABELS:
            f.write('item { \n')
            f.write('\tname:\'{}\'\n'.format(label['name']))
            f.write('\tid:{}\n'.format(label['id']))
            f.write('}\n')

    # Create TF records from train images
    cmd = "python " + files['TF_RECORD_SCRIPT'] + " -x " + os.path.join(paths['IMAGE_PATH'], 'train') + " -l " + files['LABELMAP'] + " -o " + os.path.join(paths['ANNOTATION_PATH'], 'train.record')
    os.system(cmd)

    # Create TF records from test images
    cmd = "python " + files['TF_RECORD_SCRIPT'] + " -x " + os.path.join(paths['IMAGE_PATH'], 'test') + " -l " + files['LABELMAP'] + " -o " + os.path.join(paths['ANNOTATION_PATH'], 'test.record')
    os.system(cmd)

    # Remove old model
    cmd = "rm " + "tensorflow/workspace/models/" + CUSTOM_MODEL_NAME + " -r"
    os.system(cmd)

    # Create model folder
    cmd = "mkdir " + "tensorflow/workspace/models/" + CUSTOM_MODEL_NAME
    os.system(cmd)

    # Copy model config to training folder
    cmd = "cp " + os.path.join(paths['PRETRAINED_MODEL_PATH'], PRETRAINED_MODEL_NAME, 'pipeline.config') + " " + os.path.join(paths['CHECKPOINT_PATH'], 'pipeline.config')
    os.system(cmd)

    # Update config for transfer learning
    # config = config_util.get_configs_from_pipeline_file(files['PIPELINE_CONFIG'])
    pipeline_config = pipeline_pb2.TrainEvalPipelineConfig()
    with tf.io.gfile.GFile(files['PIPELINE_CONFIG'], "r") as f:                                                                                                                                                                                                                     
        proto_str = f.read()                                                                                                                                                                                                                                          
        text_format.Merge(proto_str, pipeline_config)
    
    pipeline_config.model.ssd.num_classes = len(LABELS)
    pipeline_config.train_config.batch_size = 4
    pipeline_config.train_config.fine_tune_checkpoint = os.path.join(paths['PRETRAINED_MODEL_PATH'], PRETRAINED_MODEL_NAME, 'checkpoint', 'ckpt-0')
    pipeline_config.train_config.fine_tune_checkpoint_type = "detection"
    pipeline_config.train_input_reader.label_map_path= files['LABELMAP']
    pipeline_config.train_input_reader.tf_record_input_reader.input_path[:] = [os.path.join(paths['ANNOTATION_PATH'], 'train.record')]
    pipeline_config.eval_input_reader[0].label_map_path = files['LABELMAP']
    pipeline_config.eval_input_reader[0].tf_record_input_reader.input_path[:] = [os.path.join(paths['ANNOTATION_PATH'], 'test.record')]

    config_text = text_format.MessageToString(pipeline_config)                                                                                                                                                                                                        
    with tf.io.gfile.GFile(files['PIPELINE_CONFIG'], "wb") as f:                                                                                                                                                                                                                     
        f.write(config_text)
        