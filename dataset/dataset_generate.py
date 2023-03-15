###################################################################################################
# Title: Generate dataset
# Description: Generate all necessary dataset files from images for model training
# 
# Jure Rebernik magistrska naloga
###################################################################################################

import os

if __name__ == '__main__':
    # Create TF records from train images
    os.system('rm tensorflow/workspace/annotations/train.record')
    cmd = 'python tensorflow/scripts/generate_tfrecord.py -x dataset/images/train -l tensorflow/workspace/annotations/label_map.pbtxt -o tensorflow/workspace/annotations/train.record'
    # cmd = "python " + files['TF_RECORD_SCRIPT'] + " -x " + os.path.join(paths['IMAGE_PATH'], 'train') + " -l " + files['LABELMAP'] + " -o " + os.path.join(paths['ANNOTATION_PATH'], 'train.record')
    os.system(cmd)

    # Create TF records from test images
    os.system('rm tensorflow/workspace/annotations/test.record')
    # cmd = "python " + files['TF_RECORD_SCRIPT'] + " -x " + os.path.join(paths['IMAGE_PATH'], 'test') + " -l " + files['LABELMAP'] + " -o " + os.path.join(paths['ANNOTATION_PATH'], 'test.record')
    cmd = 'python tensorflow/scripts/generate_tfrecord.py -x dataset/images/test -l tensorflow/workspace/annotations/label_map.pbtxt -o tensorflow/workspace/annotations/test.record'
    os.system(cmd)
