# Copyright 2020 Huy Le Nguyen (@usimarit)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

import logging
logging.getLogger("tensorflow").setLevel(logging.INFO)

import tensorflow as tf
from tqdm import tqdm

from tensorflow_asr.metrics.error_rates import ErrorRate
from tensorflow_asr.utils.file_util import read_file
from tensorflow_asr.utils.metric_util import cer, wer

logger = tf.get_logger()

import fire
#import tensorflow as tf
from tensorflow_asr.utils.file_util import read_file
import json
from pydub import AudioSegment

from tensorflow_asr.featurizers.speech_featurizers import read_raw_audio

from datetime import datetime
import argparse

def writeListFile(file_path, output_list, encoding = None):
    f = open(file_path, mode = "w")
    output_str = "\n".join(output_list)
    f.write(output_str)
    f.close()

def readFile(file_path, encoding = None):
    f = open(file_path, 'r')
    lines = f.read().splitlines()
#    print(lines[0])
    res = []
    for line in lines:
        line = line.strip()
        res.append(line)
    f.close()
    return res

def compare_normal_output(filepath: str):

    logger.info(f"Evaluating result from {filepath} ...")
    metrics = {
        "wer": ErrorRate(wer, name="wer", dtype=tf.float32),
        "cer": ErrorRate(cer, name="cer", dtype=tf.float32),
    }
    with read_file(filepath) as path:
        with open(path, "r", encoding="utf-8") as openfile:
            lines = openfile.read().splitlines()
    for eachline in tqdm(lines):
        groundtruth, asr_output = eachline.split("\t")
        groundtruth = tf.convert_to_tensor([groundtruth], dtype=tf.string)
        asr_output = tf.convert_to_tensor([asr_output], dtype=tf.string)
        metrics["wer"].update_state(decode=asr_output, target=groundtruth)
        metrics["cer"].update_state(decode=asr_output, target=groundtruth)
    for key, value in metrics.items():
        logger.info(f"{key}: {value.result().numpy()}")

def main(
    tflite: str,
    #audios_path: str,
    #true_text_path: str,
    test_data_file: str,
    output: str,
    blank: int = 0,
    num_rnns: int = 1,
    nstates: int = 2,
#    statesize: int = 320,
    statesize: int = 640,
):

    # concatenated_ss_audios_path = audios_path
    # print("processing audios in %s" % concatenated_ss_audios_path)

    # asr_true_text_path = true_text_path
    # asr_true_lines = readFile(asr_true_text_path)

    with open(test_data_file) as f:
        asr_true_lines = f.readlines()

    tflitemodel = tf.lite.Interpreter(model_path=tflite)    
    input_details = tflitemodel.get_input_details()
    print("input_details: ")
    print(input_details)

    output_details = tflitemodel.get_output_details()
    print("output_details: ")
    print(output_details)
    
    evaluate_lines = []
    for idx, true_line in enumerate(tqdm(asr_true_lines[1:])): # skip the header
        segments = true_line.split('\t')
        audio_f = segments[0].strip()
        filename = audio_f
        truth = segments[2].strip()

        signal = read_raw_audio(filename)
        print(type(signal))
        print(signal.shape)
        tflitemodel.resize_tensor_input(input_details[2]["index"], signal.shape)
        tflitemodel.allocate_tensors()
        tflitemodel.set_tensor(input_details[2]["index"], signal)
        tflitemodel.set_tensor(input_details[1]["index"], tf.constant(blank, dtype=tf.int32))
        tflitemodel.set_tensor(input_details[0]["index"], tf.zeros([num_rnns, nstates, 1, statesize], dtype=tf.float32))
        tflitemodel.invoke()
        hyp0 = tflitemodel.get_tensor(output_details[0]["index"])
        hyp1 = tflitemodel.get_tensor(output_details[1]["index"])
        hyp2 = tflitemodel.get_tensor(output_details[2]["index"])
    
        if idx == 0:    
            print("hyp0", hyp0)
            print("hyp1", hyp1)
            print("hyp2", hyp2)

        pred_line = "".join([chr(u) for u in hyp1])
        line_to_compare = truth + "\t" + pred_line.strip().lower()
        evaluate_lines.append(line_to_compare)
       
    writeListFile(output, evaluate_lines)
    print("write %s lines to %s" % (len(evaluate_lines), output))
    
    compare_normal_output(output)
        
if __name__ == "__main__":

    time_s = datetime.now()

    parser = argparse.ArgumentParser(description = "control the functions for evaluating tflite with files")
    parser.add_argument("--tflite_model", action='store', type=str, default = "./tflite_models/pretrained_librispeech_train_ss_test_concatenated_epoch50_noOptimize.tflite", required=True)
    #parser.add_argument("--audios_path", action='store', type=str, default = "/home/liuyi/audio_data/sample100/signs_symptoms_audio_concatenated", required=True)
    #parser.add_argument("--true_text_path", action='store', type=str, default = "/home/liuyi/tflite_experimental/emsBert/data/text_for_audio_data/liuyi/sampled_signs_symptoms_100_no_label_blank_separator.txt", required=True)
    #parser.add_argument("--data", action='store', type=str, default = None, required=True)
    #parser.add_argument("--output", action='store', type=str, default = "./test_outputs/evaluate_liuyitflite_to_delete.txt", required=True)
    args = parser.parse_args()

    print()
    print("args.tflite_model =", args.tflite_model)
    data_path = '/'.join(args.tflite_model.split('/')[:-4]) + '/finetune_test.tsv'
    print("data_path =", data_path)

    output_path = '/'.join(args.tflite_model.split('/')[:-1]) + '/tflite_test_output.tsv'
    print("output_path =", output_path)
    print()
    #stop

    main(tflite=args.tflite_model,
         test_data_file = data_path,
         #audios_path=args.audios_path,
         #true_text_path=args.true_text_path,
         output=output_path)

    time_t = datetime.now() - time_s
    print("This run takes %s" % time_t)

